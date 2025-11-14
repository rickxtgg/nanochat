"""
基础模型损失评估和采样脚本

功能说明：
本脚本用于加载训练好的基础模型检查点，并执行两项任务：
1. 损失评估：在训练集和验证集的较大数据块上评估模型损失（bits per byte）
2. 文本采样：使用多个提示词生成文本样本，直观观察模型能力

与训练过程中的快速评估不同，本脚本：
- 使用更多token进行更准确的损失估计
- 不需要训练环境，可以独立运行
- 适合在训练完成后进行全面评估

运行方式：

1. 单GPU评估：
   python -m scripts.base_loss

2. 多GPU分布式评估（推荐）：
   torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
   说明：分布式运行可以加速评估过程

3. 自定义参数：
   python -m scripts.base_loss --device_batch_size=64 --split_tokens=40960000

输出：
- 训练集和验证集的 bpb (bits per byte) 损失
- 多个提示词的生成样本
- 记录到实验报告系统
"""
import os  # 文件和目录操作
from contextlib import nullcontext  # 空上下文管理器
import torch  # PyTorch深度学习框架
from nanochat.checkpoint_manager import load_model  # 模型检查点加载
from nanochat.common import compute_init, print0, compute_cleanup, autodetect_device_type  # 通用工具函数
from nanochat.dataloader import tokenizing_distributed_data_loader  # 分布式数据加载器
from nanochat.tokenizer import get_token_bytes  # 获取token字节映射
from nanochat.loss_eval import evaluate_bpb  # 损失评估（bits per byte）
from nanochat.engine import Engine  # 文本生成引擎

# =============================================================================
# 配置参数
# =============================================================================
device_batch_size = 32  # 每个设备的批次大小
split_tokens = 20*524288  # 每个数据分片（训练集/验证集）评估的token数量
model_tag = None  # 可选：模型标签（用于指定检查点目录）
model_step = None  # 可选：模型步数（用于加载特定步数的检查点）
device_type = ""  # 设备类型：cuda|cpu|mps（空值表示自动检测）
# 允许通过命令行或配置文件覆盖上述配置
exec(open(os.path.join('nanochat', 'configurator.py')).read())

# =============================================================================
# 加载模型和分词器
# =============================================================================
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=model_step)
sequence_len = meta["model_config"]["sequence_len"]  # 序列长度（可以是任意值）
# 设置混合精度上下文（CUDA使用bfloat16）
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# =============================================================================
# 评估损失（训练集和验证集）
# =============================================================================
tokens_per_step = device_batch_size * sequence_len * ddp_world_size
assert split_tokens % tokens_per_step == 0, "split_tokens 必须能被 tokens_per_step 整除"
steps = split_tokens // tokens_per_step  # 计算评估步数
token_bytes = get_token_bytes(device=device)  # 获取token到字节的映射
bpb_results = {}

for split_name in ["train", "val"]:
    # 为每个数据分片创建数据加载器
    loader = tokenizing_distributed_data_loader(device_batch_size, sequence_len, split_name, device=device)
    with autocast_ctx:
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
    split_name_cn = "训练集" if split_name == "train" else "验证集"
    print0(f"{split_name_cn} bpb: {bpb:.4f}")
    bpb_results[split_name] = bpb

# =============================================================================
# 主进程：从模型采样生成文本
# =============================================================================
samples = []
if ddp_rank == 0:
    # 预定义的提示词列表
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        tokens = tokenizer(prompt, prepend="<|bos|>")  # 添加BOS标记
        with autocast_ctx:
            # temperature=0 表示贪婪解码（确定性生成）
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
        sample_str = tokenizer.decode(sample[0])
        print0(sample_str)
        samples.append(sample_str)

# =============================================================================
# 记录到实验报告
# =============================================================================
from nanochat.report import get_report
get_report().log(section="Base model loss", data=[
    {
        "train bpb": bpb_results["train"],
        "val bpb": bpb_results["val"],
    },
    {f"sample {i}": sample for i, sample in enumerate(samples)},
])

# =============================================================================
# 清理资源
# =============================================================================
compute_cleanup()
