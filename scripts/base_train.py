"""
基础模型训练脚本

这是nanochat的主要预训练脚本，用于从头训练GPT模型。

运行方式：

单GPU训练：
    python -m scripts.base_train

分布式训练（多GPU）：
    torchrun --nproc_per_node=8 -m scripts.base_train

CPU/MacBook训练（小模型示例）：
    python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 \
           --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20

核心功能：
    1. 模型训练：从随机权重开始训练GPT模型
    2. 梯度累积：支持大批次训练（通过累积小批次）
    3. 混合优化器：AdamW（embedding/lm_head）+ Muon（Transformer块）
    4. 学习率调度：warmup + 恒定 + warmdown
    5. 验证评估：定期评估验证集困惑度（bpb）
    6. CORE指标：定期评估核心任务性能
    7. 检查点保存：支持训练恢复
    8. WandB日志：可选的实验跟踪

训练超参数：
    - depth: 模型深度（决定层数和维度）
    - max_seq_len: 最大序列长度
    - total_batch_size: 总批次大小（tokens）
    - learning rates: embedding_lr, unembedding_lr, matrix_lr
    - training horizon: num_iterations或target_flops或target_param_data_ratio

支持的设备：
    - CUDA（推荐，支持DDP和混合精度）
    - MPS（Apple Silicon）
    - CPU（仅限小模型测试）
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# =============================================================================
# 用户配置（可通过命令行参数覆盖）
# =============================================================================

# --- WandB运行名称 ---
run = "dummy"  # WandB运行名称（"dummy"表示不记录到WandB）

# --- 运行时设置 ---
device_type = ""  # 设备类型：cuda|mps|cpu（空=自动检测，优先级：CUDA > MPS > CPU）

# --- 模型架构 ---
depth = 20  # Transformer深度（决定层数和维度，其他参数自动推导）
max_seq_len = 2048  # 最大上下文长度（序列长度）

# --- 训练时长（3种方式，优先级从高到低） ---
num_iterations = -1  # 方式1：明确指定优化步数（-1 = 禁用）
target_flops = -1.0  # 方式2：根据目标FLOPs计算步数（适合缩放定律实验，-1 = 禁用）
target_param_data_ratio = 20  # 方式3：维持固定的数据:参数比（Chinchilla=20，-1 = 禁用）

# --- 优化器设置 ---
device_batch_size = 32  # 每设备批次大小（根据显存设置，避免OOM）
total_batch_size = 524288  # 总批次大小（tokens数）
embedding_lr = 0.2  # embedding参数的学习率（AdamW）
unembedding_lr = 0.004  # lm_head参数的学习率（AdamW）
weight_decay = 0.0  # embedding/lm_head的权重衰减（AdamW）
matrix_lr = 0.02  # Transformer块参数的学习率（Muon）
grad_clip = 1.0  # 梯度裁剪值（0.0 = 禁用）

# --- 学习率调度 ---
warmup_ratio = 0.0  # 学习率预热比例（相对于总步数）
warmdown_ratio = 0.2  # 学习率衰减比例（相对于总步数）
final_lr_frac = 0.0  # 最终学习率相对初始学习率的比例

# --- 训练恢复 ---
resume_from_step = -1  # 从指定步数恢复训练（-1 = 禁用，从头开始）

# --- 评估设置 ---
eval_every = 250  # 每隔多少步评估验证集bpb
eval_tokens = 20*524288  # 评估验证损失时使用的token数
core_metric_every = 2000  # 每隔多少步评估CORE指标（-1 = 禁用）
core_metric_max_per_task = 500  # 评估CORE指标时每任务的样本数
sample_every = 2000  # 每隔多少步采样模型输出
save_every = -1  # 每隔多少步保存检查点（-1 = 禁用，仅在结束时保存）

# --- 输出设置 ---
model_tag = ""  # 可选：覆盖输出检查点目录的模型标签名

# 配置系统：允许通过命令行或配置文件覆盖上述设置
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())  # 从命令行或配置文件读取覆盖
user_config = {k: globals()[k] for k in config_keys}  # 保存最终配置（用于日志记录）

# =============================================================================
# 计算环境初始化
# =============================================================================

# 初始化计算环境
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # 主进程负责日志记录、检查点保存等
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# WandB日志初始化
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# =============================================================================
# 分词器和词汇表
# =============================================================================

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"词汇表大小: {vocab_size:,}")

# =============================================================================
# 模型架构配置
# =============================================================================

# 根据depth推导模型参数
num_layers = depth
model_dim = depth * 64  # 宽高比64（通常随模型增大从64变到128）
num_heads = max(1, (model_dim + 127) // 128)  # 头维度128（这里是向上取整除法）
num_kv_heads = num_heads  # 默认1:1 GQA比例（即禁用GQA）
print0(f"层数: {num_layers}")
print0(f"模型维度: {model_dim}")
print0(f"Query头数: {num_heads}")
print0(f"KV头数: {num_kv_heads}")

# =============================================================================
# 批次大小和梯度累积
# =============================================================================

# 计算达到目标总批次大小所需的梯度累积步数
tokens_per_fwdbwd = device_batch_size * max_seq_len  # 单个rank每次前向/反向的token数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # 所有rank每次前向/反向的总token数
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / 微批次 / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / 微批次（全局）: {world_tokens_per_fwdbwd:,}")
print0(f"总批次大小 {total_batch_size:,} => 梯度累积步数: {grad_accum_steps}")

# =============================================================================
# 模型初始化
# =============================================================================

# 创建新模型（随机权重）
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim
)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# 如果恢复训练，用检查点参数覆盖模型参数
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}"  # 例如：d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
if resuming:
    print0(f"从步数 {resume_from_step} 恢复优化")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data  # 复制后释放内存

orig_model = model  # 保存原始未编译模型（用于保存state_dict和推理/评估，因为形状可能变化）
model = torch.compile(model, dynamic=False)  # 编译模型（输入形状固定，可以用dynamic=False）
num_params = sum(p.numel() for p in model.parameters())
print0(f"参数数量: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"每token估计FLOPs: {num_flops_per_token:e}")

# =============================================================================
# 训练时长计算
# =============================================================================

# 计算迭代次数（3种方式，按优先级）
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"使用用户指定的迭代次数: {num_iterations:,}")
elif target_flops > 0:
    # 从目标FLOPs计算迭代次数
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"根据目标FLOPs计算的迭代次数: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # 从目标数据:参数比计算迭代次数
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"根据目标数据:参数比计算的迭代次数: {num_iterations:,}")
else:
    raise ValueError("未指定训练时长")
total_tokens = total_batch_size * num_iterations
print0(f"训练总token数: {total_tokens:,}")
print0(f"Tokens:Params比例: {total_batch_size * num_iterations / num_params:.2f}")  # Chinchilla约为20
print0(f"训练总FLOPs估计: {num_flops_per_token * total_tokens:e}")

# =============================================================================
# 优化器初始化
# =============================================================================

# 混合优化器：Muon（线性层）+ AdamW（embedding和lm_head）
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay
)
adamw_optimizer, muon_optimizer = optimizers

# 恢复优化器状态（如果恢复训练）
if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data  # 释放内存

# =============================================================================
# 数据加载器初始化
# =============================================================================

tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(
    device_batch_size, max_seq_len, split="train", device=device,
    resume_state_dict=dataloader_resume_state_dict
)
build_val_loader = lambda: tokenizing_distributed_data_loader(
    device_batch_size, max_seq_len, split="val", device=device
)
x, y, dataloader_state_dict = next(train_loader)  # 启动加载第一批数据

# =============================================================================
# 超参数调度器
# =============================================================================

# 学习率调度器
def get_lr_multiplier(it):
    """
    学习率乘数调度器（warmup -> 恒定 -> warmdown）
    
    参数：
        it: 当前迭代步数
    
    返回：
        lr_multiplier: 学习率乘数（相对于初始学习率）
    
    调度策略：
        1. Warmup阶段：从0线性增长到1
        2. 恒定阶段：保持1
        3. Warmdown阶段：从1线性衰减到final_lr_frac
    """
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        # Warmup: 线性增长
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        # 恒定阶段
        return 1.0
    else:
        # Warmdown: 线性衰减
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Muon优化器动量调度器
def get_muon_momentum(it):
    """
    Muon优化器动量调度器
    
    参数：
        it: 当前迭代步数
    
    返回：
        momentum: 动量值
    
    调度策略：
        前300步从0.85线性增长到0.95，之后保持0.95
    """
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# =============================================================================
# 训练循环状态
# =============================================================================

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # 训练损失的指数移动平均（EMA）
    total_training_time = 0  # 总训练时间（墙钟时间）
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# =============================================================================
# 主训练循环
# =============================================================================

while True:
    last_step = step == num_iterations  # 循环运行num_iterations+1次，以便在最后评估/保存
    flops_so_far = num_flops_per_token * total_batch_size * step

    # ===== 定期评估验证集bpb（所有rank参与） =====
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"步数 {step:05d} | 验证bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # ===== 定期评估CORE指标（所有rank参与） =====
    # 使用原始未编译模型（因为输入形状不断变化）
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"步数 {step:05d} | CORE指标: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # ===== 定期采样模型输出（仅主进程） =====
    # 使用原始未编译模型（因为输入形状不断变化）
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)  # 使用orig_model避免重新编译
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # ===== 保存检查点 =====
    # 在运行结束时保存，或每隔save_every步保存（除了第一步或恢复步）
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),  # 模型参数
            [opt.state_dict() for opt in optimizers],  # 优化器状态
            {  # 元数据（保存为JSON）
                "step": step,
                "val_bpb": val_bpb,  # 最后一步的损失
                "model_config": model_config_kwargs,
                "user_config": user_config,  # 训练脚本的输入配置
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {  # 所有循环状态（除step外），用于恢复训练
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # ===== 终止条件 =====
    # TODO: 可能还需要添加损失爆炸等条件
    if last_step:
        break

    # =========================================================================
    # 单步训练
    # =========================================================================
    
    # 计算梯度
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()  # 用于日志记录
        loss = loss / grad_accum_steps  # 每次.backward()是梯度求和 => 这里归一化损失
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)  # GPU忙于前向/反向时预取下一批数据
    
    # 梯度裁剪
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item()  # GPU张量 -> CPU浮点数（注意：CPU-GPU同步点）
    
    # 更新优化器
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # =========================================================================

    # ===== 日志记录 =====
    ema_beta = 0.9  # EMA衰减因子（用于平滑日志输出）
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()  # 训练损失的EMA
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))  # EMA去偏
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size  # bfloat16 H100 SXM且不使用2:4稀疏性
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # MFU（模型FLOPs利用率），单位%
    if step > 10:
        total_training_time += dt  # 只统计前10步之后的时间
    print_grad_norm = f" 梯度范数: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print0(f"步数 {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | 损失: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | 总时间: {total_training_time/60:.2f}分钟")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)

    # 状态更新
    step += 1

# =============================================================================
# 训练完成统计
# =============================================================================

print0(f"峰值内存使用: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"总训练时间: {total_training_time/60:.2f}分钟")
print0(f"最小验证bpb: {min_val_bpb:.4f}")

# 记录到报告
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config,  # CLI参数
    {  # 训练设置统计
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    {  # 训练结果统计
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# =============================================================================
# 清理
# =============================================================================

wandb_run.finish()  # 完成WandB运行
compute_cleanup()  # 清理计算环境（销毁DDP进程组等）