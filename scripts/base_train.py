"""
基础模型训练脚本 - nanochat 预训练阶段

功能说明：
本脚本实现大语言模型的基础预训练（Base Pre-training），从零开始训练一个 Transformer 模型。
这是整个训练流程的第一阶段，模型在大规模文本语料上学习语言的统计规律。

运行方式：

1. 单GPU训练：
   python base_train.py

2. 多GPU分布式训练（推荐）：
   torchrun --nproc_per_node=8 base_train.py
   说明：使用 PyTorch 的 DistributedDataParallel (DDP) 进行数据并行训练

3. CPU/Macbook 演示训练（小模型）：
   python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
   说明：训练一个非常小的4层模型，仅用于代码验证，无法获得实用效果

主要特性：
- 支持混合精度训练（bfloat16）
- 支持分布式数据并行（DDP）
- 使用 Muon 优化器优化矩阵参数，AdamW 优化 embedding 参数
- 支持梯度累积以实现大批次训练
- 支持学习率预热(warmup)和冷却(warmdown)
- 定期评估验证集损失和 CORE 指标
- 定期采样生成文本，监控训练进度
- 集成 Weights & Biases (WandB) 进行实验跟踪
"""

import os
# 设置 PyTorch CUDA 内存分配策略：使用可扩展段，减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time  # 用于训练时间统计
from contextlib import nullcontext  # 在非CUDA设备上提供空的上下文管理器

import wandb  # Weights & Biases 实验跟踪工具
import torch  # PyTorch 深度学习框架

from nanochat.gpt import GPT, GPTConfig  # GPT 模型定义和配置
from nanochat.dataloader import tokenizing_distributed_data_loader  # 分布式数据加载器
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type  # 通用工具函数
from nanochat.tokenizer import get_tokenizer, get_token_bytes  # 分词器相关
from nanochat.checkpoint_manager import save_checkpoint  # 检查点保存管理
from nanochat.loss_eval import evaluate_bpb  # 损失评估（每字节比特数）
from nanochat.engine import Engine  # 文本生成引擎
from scripts.base_eval import evaluate_model  # 模型评估（CORE指标）
print_banner()  # 打印 nanochat 项目横幅

# =============================================================================
# 用户配置参数 - 可通过命令行参数覆盖
# =============================================================================

# -------------------- 实验跟踪配置 --------------------
run = "dummy"  # WandB 运行名称（"dummy" 为特殊值，表示不上传到 WandB）

# -------------------- 运行环境配置 --------------------
device_type = ""  # 设备类型：cuda|cpu|mps（空值表示自动检测，优先级：CUDA > MPS > CPU）

# -------------------- 模型架构配置 --------------------
depth = 20  # Transformer 模型深度（层数），其他参数将根据此值自动推导
max_seq_len = 2048  # 最大上下文长度（tokens）

# -------------------- 训练时长配置（三选一，按优先级排序）--------------------
num_iterations = -1  # 显式指定优化步数（-1 表示禁用此选项）
target_flops = -1.0  # 根据目标FLOPs计算迭代次数，用于缩放律实验（-1 表示禁用）
target_param_data_ratio = 20  # 根据固定的数据:参数比例计算迭代次数（Chinchilla论文建议=20，-1 表示禁用）

# -------------------- 优化器配置 --------------------
device_batch_size = 32  # 每个设备的批次大小（设置为不会OOM的最大值）
total_batch_size = 524288  # 总期望批次大小（以token数计算）
embedding_lr = 0.2  # Embedding 参数的学习率（使用 AdamW 优化器）
unembedding_lr = 0.004  # Unembedding（输出层）参数的学习率（使用 AdamW 优化器）
weight_decay = 0.0  # Embedding/Unembedding 参数的权重衰减（L2正则化）
matrix_lr = 0.02  # 矩阵参数的学习率（使用 Muon 优化器）
grad_clip = 1.0  # 梯度裁剪阈值（0.0 表示禁用梯度裁剪）
warmup_ratio = 0.0  # 学习率预热阶段占总训练步数的比例
warmdown_ratio = 0.2  # 学习率冷却阶段占总训练步数的比例（最后20%步数）
final_lr_frac = 0.0  # 最终学习率占初始学习率的比例（0.0 表示降到0）

# -------------------- 评估配置 --------------------
eval_every = 250  # 每隔多少步评估一次验证集损失（bits per byte）
eval_tokens = 20*524288  # 评估验证集损失时使用的token数量
core_metric_every = 2000  # 每隔多少步评估一次 CORE 指标（-1 表示禁用）
core_metric_max_per_task = 500  # 评估 CORE 指标时每个任务最多使用的样本数
sample_every = 2000  # 每隔多少步从模型采样生成文本

# -------------------- 输出配置 --------------------
model_tag = ""  # 可选：覆盖输出检查点目录的模型标签名称（默认使用 d{depth}）

# -------------------- 命令行参数覆盖机制 --------------------
# 收集所有可配置的参数（排除私有变量和非基础类型）
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# 通过 configurator.py 允许命令行或配置文件覆盖上述设置
exec(open(os.path.join('nanochat', 'configurator.py')).read())
# 保存用户配置，用于日志记录和实验追踪
user_config = {k: globals()[k] for k in config_keys}
# =============================================================================

# =============================================================================
# 计算环境初始化
# =============================================================================
# 自动检测或使用指定的设备类型
device_type = autodetect_device_type() if device_type == "" else device_type
# 初始化分布式训练环境（DDP）或单机训练环境
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # 主进程负责日志记录、检查点保存等
# 设置混合精度训练上下文（CUDA使用bfloat16，其他设备使用默认精度）
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
# 同步函数：确保所有CUDA操作完成（用于精确计时）
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
# 获取最大GPU内存使用量的函数
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# =============================================================================
# Weights & Biases 日志记录初始化
# =============================================================================
# 使用虚拟 WandB 的条件：运行名为 "dummy" 或非主进程
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# =============================================================================
# 分词器初始化
# =============================================================================
# 分词器用于文本生成和评估，同时需要获取词汇表大小以构建模型
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)  # 每个token对应的平均字节数，用于 bpb 计算
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# =============================================================================
# 模型架构参数推导
# =============================================================================
# 根据用户指定的深度(depth)自动计算其他架构参数
num_layers = depth
# 模型维度 = 深度 × 64（纵横比为64，随着模型增大通常从64变化到128）
model_dim = depth * 64
# 注意力头数量：确保每个头的维度为128（向上取整除法）
num_heads = max(1, (model_dim + 127) // 128)
# KV头数量：默认1:1比例（即GQA分组查询注意力机制未启用）
num_kv_heads = num_heads
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# =============================================================================
# 优化器和数据相关的超参数计算
# =============================================================================
# 计算达到目标总批次大小所需的梯度累积步数
tokens_per_fwdbwd = device_batch_size * max_seq_len  # 单个设备每次前向+反向传播处理的token数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # 所有设备每次迭代处理的总token数
assert total_batch_size % world_tokens_per_fwdbwd == 0, "总批次大小必须能被每次迭代的总token数整除"
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd  # 梯度累积步数
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# =============================================================================
# 模型初始化
# =============================================================================
# 构建模型配置参数字典
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
# 在 "meta" 设备上创建模型结构（不分配实际内存）
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
# 将模型转移到目标设备并分配内存（但参数仍未初始化）
model.to_empty(device=device)
# 初始化模型权重（使用特定的初始化策略）
model.init_weights()
# 保存原始未编译的模型引用，用于保存检查点时获取原始 state_dict
orig_model = model
# 使用 PyTorch 2.0 的编译功能加速训练（dynamic=False 表示固定输入形状）
model = torch.compile(model, dynamic=False)
# 统计模型参数总数
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
# 估算每个token的计算量（FLOPs）
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# =============================================================================
# 训练迭代次数计算
# =============================================================================
# 三种方式计算训练步数（按优先级）：显式指定、目标FLOPs、目标数据参数比
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0, "必须指定训练时长"
if num_iterations > 0:
    # 方式1：使用用户明确指定的迭代次数
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # 方式2：根据目标计算量(FLOPs)反推迭代次数
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # 方式3：根据目标数据参数比计算迭代次数（Chinchilla论文建议比例约为20）
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")

# 计算并打印训练统计信息
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}")  # Chinchilla建议约20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# =============================================================================
# 优化器初始化
# =============================================================================
# 使用混合优化器策略：
# - Muon优化器：用于线性层（矩阵参数）
# - AdamW优化器：用于 embedding 和 lm_head（输出层）
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

# =============================================================================
# 数据加载器初始化
# =============================================================================
base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")
# 训练集数据加载器（支持分布式采样）
train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train", device=device)
# 验证集数据加载器构建函数（延迟创建，每次评估时重新创建）
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
# 预加载第一批数据，启动异步数据加载流水线
x, y = next(train_loader)

# =============================================================================
# 超参数调度器设置
# =============================================================================

def get_lr_multiplier(it):
    """
    学习率调度器：实现预热(warmup)和冷却(warmdown)
    
    训练分为三个阶段：
    1. 预热阶段：学习率从0线性增加到1.0
    2. 稳定阶段：学习率保持1.0
    3. 冷却阶段：学习率从1.0线性衰减到 final_lr_frac
    
    参数：
        it: 当前迭代步数
        
    返回：
        学习率乘数（相对于初始学习率）
    """
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        # 预热阶段：线性增长
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        # 稳定阶段：保持1.0
        return 1.0
    else:
        # 冷却阶段：线性衰减
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

def get_muon_momentum(it):
    """
    Muon优化器的动量调度器
    
    动量在前300步从0.85线性增加到0.95
    
    参数：
        it: 当前迭代步数
        
    返回：
        动量值
    """
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# =============================================================================
# 主训练循环
# =============================================================================
min_val_bpb = float("inf")  # 记录最小验证集损失
smooth_train_loss = 0  # 训练损失的指数移动平均(EMA)
ema_beta = 0.9  # EMA衰减因子
total_training_time = 0  # 总训练时间（墙钟时间）

# 注意：循环次数为 num_iterations + 1，以便在最后一步进行评估和保存
for step in range(num_iterations + 1):
    last_step = step == num_iterations  # 是否为最后一步
    flops_so_far = num_flops_per_token * total_batch_size * step  # 累积的计算量

    # =========================================================================
    # 定期评估：验证集损失（所有进程参与）
    # =========================================================================
    if last_step or step % eval_every == 0:
        model.eval()  # 切换到评估模式
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()  # 切换回训练模式

    # =========================================================================
    # 定期评估：CORE 指标（所有进程参与）
    # =========================================================================
    # 使用原始未编译模型，因为输入形状会变化
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # =========================================================================
    # 定期采样：从模型生成文本（仅主进程执行）
    # =========================================================================
    # 使用原始未编译模型，因为输入形状会变化
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        # 预定义的提示词列表，用于观察模型的生成能力
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)  # 使用原始模型避免重新编译
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                # temperature=0 表示贪婪解码（确定性生成）
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # =========================================================================
    # 保存检查点（仅主进程，仅在最后一步）
    # =========================================================================
    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}"  # 例如：d20
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),  # 保存原始模型的权重
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,  # 最后一步的验证集损失
                "model_config": model_config_kwargs,
                "user_config": user_config,  # 训练脚本的输入参数
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            }
        )

    if last_step:
        break  # 最后一步后退出循环

    # =========================================================================
    # 单步训练：梯度计算
    # =========================================================================
    synchronize()  # 同步所有设备，确保计时准确
    t0 = time.time()
    
    # 梯度累积循环：将大批次分解为多个小批次处理
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)  # 前向传播计算损失
        train_loss = loss.detach()  # 保存用于日志记录（分离计算图）
        loss = loss / grad_accum_steps  # 归一化损失（因为.backward()累加梯度）
        loss.backward()  # 反向传播计算梯度
        x, y = next(train_loader)  # 预取下一批数据（与GPU计算并行）
    
    # 梯度裁剪：防止梯度爆炸
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item()  # GPU张量 -> CPU浮点数（注意：这是一个同步点）
    
    # 更新优化器：应用学习率调度
    lrm = get_lr_multiplier(step)  # 获取当前学习率乘数
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm  # 更新学习率
    
    # 更新 Muon 优化器的动量
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    
    # 执行优化器步骤
    for opt in optimizers:
        opt.step()
    
    # 清空梯度（set_to_none=True 更高效）
    model.zero_grad(set_to_none=True)
    
    synchronize()  # 同步所有设备
    t1 = time.time()
    dt = t1 - t0  # 单步训练耗时
    # =========================================================================

    # =========================================================================
    # 训练日志记录
    # =========================================================================
    # 计算训练损失的指数移动平均（平滑曲线）
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    # 去偏的EMA（修正初始偏差）
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    
    # 计算训练统计指标
    pct_done = 100 * step / num_iterations  # 训练进度百分比
    tok_per_sec = int(total_batch_size / dt)  # 每秒处理的token数
    flops_per_sec = num_flops_per_token * total_batch_size / dt  # 每秒计算量
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size  # H100 SXM理论峰值(bfloat16, 无2:4稀疏)
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # 模型FLOPs利用率(%)
    
    # 累计训练时间（跳过前10步以排除预热影响）
    if step > 10:
        total_training_time += dt
    
    # 打印训练进度
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    
    # 每100步记录一次到 WandB
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

# =============================================================================
# 训练完成：打印最终统计信息
# =============================================================================
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# =============================================================================
# 记录到实验报告
# =============================================================================
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config,  # 命令行参数
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
# 清理资源
# =============================================================================
wandb_run.finish()  # 完成 WandB 运行
compute_cleanup()  # 清理分布式训练环境
