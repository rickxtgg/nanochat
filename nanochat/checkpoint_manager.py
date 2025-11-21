"""
检查点管理工具

这个模块提供了保存和加载模型/优化器/状态检查点的实用工具。

核心功能：
    - 保存模型权重、优化器状态和元数据
    - 从检查点加载并重建模型
    - 自动查找最大/最新的检查点
    - 支持nanochat的目录结构约定

检查点结构：
    - model_<step>.pt: 模型参数
    - optim_<step>.pt: 优化器状态（可选）
    - meta_<step>.json: 元数据（模型配置、训练信息等）

目录约定：
    - base_checkpoints/: 基础预训练模型
    - mid_checkpoints/: 中期训练模型
    - chatsft_checkpoints/: SFT模型
    - chatrl_checkpoints/: RL模型
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# =============================================================================
# 日志设置
# =============================================================================
setup_default_logging()
logger = logging.getLogger(__name__)

def log0(message):
    """
    只在主进程（rank 0）记录日志
    
    参数：
        message: 要记录的日志消息
    
    用途：
        在分布式训练中，避免多个进程重复记录相同的日志信息。
        只有rank 0（主进程）会记录日志。
    
    实现：
        通过检查环境变量RANK来判断是否为主进程。
        如果RANK未设置，默认为0（非分布式模式）。
    """
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    """
    保存检查点（模型、优化器、元数据）
    
    参数：
        checkpoint_dir: 检查点保存目录
        step: 当前训练步数
        model_data: 模型状态字典
        optimizer_data: 优化器状态字典（可以是None）
        meta_data: 元数据字典（将保存为JSON）
        rank: 当前进程的rank（默认0）
    
    文件命名格式：
        - model_{step:06d}.pt: 模型参数（仅rank 0保存）
        - meta_{step:06d}.json: 元数据（仅rank 0保存）
        - optim_{step:06d}_rank{rank}.pt: 优化器状态（每个rank保存自己的）
    
    注意：
        优化器状态在DDP模式下会跨rank分片，所以每个rank必须保存自己的优化器状态。
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        # 保存模型状态参数
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"已保存模型参数至: {model_path}")
        # 保存元数据字典为JSON
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"已保存元数据至: {meta_path}")
    # 注意：优化器状态跨rank分片，所以每个rank必须保存自己的
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"已保存优化器状态至: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    """
    加载检查点（模型、优化器、元数据）
    
    参数：
        checkpoint_dir: 检查点目录
        step: 要加载的训练步数
        device: 目标设备
        load_optimizer: 是否加载优化器状态（默认False）
        rank: 当前进程的rank（默认0）
    
    返回：
        (model_data, optimizer_data, meta_data): 元组
            - model_data: 模型状态字典
            - optimizer_data: 优化器状态字典（如果不加载则为None）
            - meta_data: 元数据字典
    
    加载的文件：
        - model_{step:06d}.pt: 模型参数
        - optim_{step:06d}_rank{rank}.pt: 优化器状态（如果load_optimizer=True）
        - meta_{step:06d}.json: 元数据
    """
    # 加载模型状态
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    
    # 如果请求，加载优化器状态
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    
    # 加载元数据
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    
    return model_data, optimizer_data, meta_data


# =============================================================================
# 模型构建和重建函数
# =============================================================================

def build_model(checkpoint_dir, step, device, phase):
    """
    从检查点构建模型
    
    这是一个便捷函数，封装了从检查点重建完整模型的重复代码。
    
    参数：
        checkpoint_dir: 检查点目录路径
        step: 训练步数
        device: 目标设备
        phase: 训练阶段，必须是"train"或"eval"
    
    返回：
        (model, tokenizer, meta_data): 元组
            - model: 未编译的基础模型（不包含DDP包装）
            - tokenizer: 分词器实例
            - meta_data: 训练时保存的元数据
    
    处理步骤：
        1. 加载检查点数据
        2. 处理设备兼容性（CPU/MPS转换bfloat16到float）
        3. 清理torch.compile产生的键名前缀
        4. 创建模型配置并实例化模型
        5. 加载模型权重
        6. 设置训练/评估模式
        7. 加载分词器并验证兼容性
    """
    assert phase in ["train", "eval"], f"无效的阶段: {phase}（必须是'train'或'eval'）"
    
    # 加载检查点数据
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    
    # CPU/MPS设备：将bfloat16张量转换为float（CPU不支持bfloat16）
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    
    # Hack: 修复torch.compile问题，它会在所有键前加上_orig_mod.前缀
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    
    # 从元数据获取模型配置
    model_config_kwargs = meta_data["model_config"]
    log0(f"正在使用配置构建模型: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    
    # 在meta设备上创建模型（不分配内存）
    with torch.device("meta"):
        model = GPT(model_config)
    
    # 加载模型状态
    model.to_empty(device=device)  # 在目标设备上分配空的张量
    model.init_weights()  # 注意：这有点笨拙，但我们需要初始化旋转嵌入。TODO: 修复模型权重重新初始化问题
    model.load_state_dict(model_data, strict=True, assign=True)  # 加载权重
    
    # 设置正确的训练阶段/模式
    if phase == "eval":
        model.eval()
    else:
        model.train()
    
    # 加载分词器
    tokenizer = get_tokenizer()
    
    # 合理性检查：验证模型和分词器的兼容性
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], \
        f"分词器词汇表大小({tokenizer.get_vocab_size()})与模型配置({model_config_kwargs['vocab_size']})不匹配"
    
    return model, tokenizer, meta_data


# =============================================================================
# 辅助函数：查找检查点
# =============================================================================

def find_largest_model(checkpoint_dir):
    """
    查找最大的模型
    
    策略：
        1. 优先查找d<number>格式的模型标签（如d12），返回数字最大的
        2. 如果没有这种格式，返回最近更新的模型
    
    参数：
        checkpoint_dir: 包含多个模型子目录的检查点目录
    
    返回：
        model_tag: 模型标签字符串（子目录名）
    
    用途：
        当用户未指定模型时，自动选择最大的模型
    """
    # 尝试猜测模型标签：选择可用的最大模型
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"未在 {checkpoint_dir} 中找到检查点")
    
    # 策略1: 通常所有模型标签都是d<number>的形式，先尝试这种
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    # 策略2: 如果失败，选择最近更新的模型
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    """
    查找最新的训练步数
    
    参数：
        checkpoint_dir: 检查点目录（包含model_<step>.pt文件）
    
    返回：
        last_step: 最大的步数
    
    实现：
        查找checkpoint_dir中所有model_*.pt文件，返回最大的步数
    
    异常：
        FileNotFoundError: 如果目录中没有检查点文件
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"未在 {checkpoint_dir} 中找到检查点文件")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# =============================================================================
# 便捷函数：考虑nanochat目录结构的高级接口
# =============================================================================

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    """
    从检查点目录加载模型（支持自动查找）
    
    参数：
        checkpoints_dir: 检查点根目录（包含多个模型子目录）
        device: 目标设备
        phase: 训练阶段（"train"或"eval"）
        model_tag: 模型标签（可选，如果不提供则自动查找最大模型）
        step: 训练步数（可选，如果不提供则使用最后一步）
    
    返回：
        (model, tokenizer, meta_data): 元组
    
    智能特性：
        - 自动查找最大模型（如果未指定model_tag）
        - 自动查找最新步数（如果未指定step）
    """
    if model_tag is None:
        # 通过默认选择最大模型来猜测模型标签
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"未提供模型标签，自动选择: {model_tag}")
    
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    
    if step is None:
        # 通过默认选择最后一步来猜测步数
        step = find_last_step(checkpoint_dir)
    
    assert step is not None, f"未在 {checkpoint_dir} 中找到检查点"
    
    # 构建模型
    log0(f"正在从 {checkpoint_dir} 加载模型（步数={step}）")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data


def load_model(source, *args, **kwargs):
    """
    从nanochat标准目录结构加载模型
    
    参数：
        source: 模型来源，必须是以下之一：
            - "base": 基础预训练模型（base_checkpoints/）
            - "mid": 中期训练模型（mid_checkpoints/）
            - "sft": SFT模型（chatsft_checkpoints/）
            - "rl": RL模型（chatrl_checkpoints/）
        *args, **kwargs: 传递给load_model_from_dir的其他参数
    
    返回：
        (model, tokenizer, meta_data): 元组
    
    示例：
        >>> model, tokenizer, meta = load_model("base", device, "eval")
        >>> model, tokenizer, meta = load_model("sft", device, "eval", model_tag="d12", step=5000)
    """
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

