"""
中期训练（Midtraining）脚本 - 在特定任务数据上继续训练基础模型

功能说明：
中期训练是介于预训练和监督微调之间的一个训练阶段。它在特定领域或任务的数据上
继续训练基础模型，使模型更好地适应目标任务，同时保持语言建模的损失函数。

中期训练 vs 预训练：
- 相似点：都使用语言建模损失（next token prediction）
- 不同点：中期训练使用更小、更有针对性的数据混合
- 数据混合：包含对话、数学、多选题等任务相关数据

训练数据混合：
1. SmolTalk: 460K行通用对话数据
2. MMLU: 100K行多选题（ARC, MC_TEST, OBQA, RACE）
3. GSM8K: 8K行数学问题
4. CustomJSON: 1K行自定义身份对话
5. SpellingBee: 600行拼写任务

运行方式：

1. 单GPU模式：
   python -m scripts.mid_train
   说明：适用于调试和小规模实验

2. 多GPU分布式训练（推荐）：
   torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
   说明：使用8个GPU并行训练，显著加速

3. 自定义参数：
   python -m scripts.mid_train --num_iterations=5000 --device_batch_size=32
   说明：指定训练步数和批次大小

技术特性：
- 混合数据训练：同时学习对话、推理、知识等多种能力
- 梯度累积：支持大批次训练
- 学习率调度：使用warmup和warmdown
- 定期评估：验证集loss和采样生成
- torch.compile：使用PyTorch编译加速
- 混合优化器：Muon用于矩阵参数，AdamW用于embedding
"""

from collections import deque  # 双端队列，用于EMA计算
import os
# 设置PyTorch CUDA内存分配策略：使用可扩展段，减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time  # 时间测量
import wandb  # Weights & Biases 实验跟踪
import torch  # PyTorch深度学习框架
from contextlib import nullcontext  # 空上下文管理器
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type  # 通用工具
from nanochat.tokenizer import get_token_bytes  # 获取token字节数
from nanochat.checkpoint_manager import save_checkpoint, load_model  # 检查点管理
from nanochat.loss_eval import evaluate_bpb  # 评估bits per byte
import torch.distributed as dist  # 分布式训练

# 导入任务数据集
from tasks.common import TaskMixture  # 任务混合工具
from tasks.gsm8k import GSM8K  # 小学数学问题
from tasks.mmlu import MMLU  # 多学科多选题
from tasks.smoltalk import SmolTalk  # 通用对话数据
from tasks.customjson import CustomJSON  # 自定义JSON对话
from tasks.spellingbee import SimpleSpelling, SpellingBee  # 拼写任务

# =============================================================================
# 中期训练超参数配置
# =============================================================================

# -------------------- 实验跟踪 --------------------
run = "dummy"  # WandB运行名称（"dummy"表示不上传到WandB）

# -------------------- 模型和设备 --------------------
device_type = ""  # 设备类型：cuda|cpu|mps（空值表示自动检测）
model_tag = None  # 要加载的模型标签
step = None  # 要加载的训练步数
dtype = "bfloat16"  # 数据类型

# -------------------- 训练参数 --------------------
num_iterations = -1  # 显式指定优化步数（-1表示禁用）
max_seq_len = 2048  # 最大序列长度
device_batch_size = 32  # 每个设备的批次大小
total_batch_size = 524288  # 总批次大小（所有GPU的总token数）

# -------------------- 优化器参数 --------------------
unembedding_lr = 0.004  # Unembedding层学习率
embedding_lr = 0.2  # Embedding层学习率
matrix_lr = 0.02  # 矩阵参数学习率（Muon优化器）
init_lr_frac = 1.0  # 初始学习率占基础学习率的比例
weight_decay = 0.0  # 权重衰减

# -------------------- 评估参数 --------------------
eval_every = 150  # 每隔多少步评估一次（-1表示禁用）
eval_tokens = 20*524288  # 评估时使用的token数

# -------------------- 其他 --------------------
dry_run = 0  # dry_run=1用于实验：记录到WandB但不保存检查点或报告

# -------------------- 命令行参数覆盖 --------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())  # 允许命令行或配置文件覆盖
user_config = {k: globals()[k] for k in config_keys}  # 保存用户配置用于日志记录
# =============================================================================

# =============================================================================
# 计算环境初始化
# =============================================================================
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # 主进程标志（用于日志和保存）
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# =============================================================================
# Weights & Biases 日志记录初始化
# =============================================================================
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-mid", name=run, config=user_config)

# =============================================================================
# 加载基础模型和分词器
# =============================================================================
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=model_tag, step=step)
pretrain_batch_size = meta.get("device_batch_size", None)
# 警告：如果当前批次大小大于预训练时的批次大小，可能导致OOM
if pretrain_batch_size is not None and device_batch_size > pretrain_batch_size:
    print0(f"⚠️ 警告: 基础模型训练时使用的批次大小为 {pretrain_batch_size}，请确认你为本脚本传入的 --device_batch_size 参数是否合适？")
orig_model = model
model = torch.compile(model, dynamic=False)  # 使用torch.compile加速
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()

# =============================================================================
# 批次大小和梯度累积计算
# =============================================================================
tokens_per_fwdbwd = device_batch_size * max_seq_len  # 单个进程每次迭代的token数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # 所有进程每次迭代的总token数
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd  # 梯度累积步数
print0(f"每个进程每次微批次的token数: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"所有进程每次微批次的总token数: {world_tokens_per_fwdbwd:,}")
print0(f"总批次大小 {total_batch_size:,} => 梯度累积步数: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# =============================================================================
# 优化器初始化
# =============================================================================
# Muon用于Linear层，AdamW用于embedding和lm_head
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

# 设置初始学习率为基础学习率的一个比例
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]  # 保存初始学习率以便后续衰减

# =============================================================================
# 中期训练数据混合和DataLoader
# =============================================================================
base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_dataset = TaskMixture([
    SmolTalk(split="train"),  # 460K行通用对话
    MMLU(subset="auxiliary_train", split="train"),  # 100K行多选题（来自ARC, MC_TEST, OBQA, RACE）
    GSM8K(subset="main", split="train"),  # 8K行数学问题（包含计算器工具使用）
    CustomJSON(filepath=identity_conversations_filepath),  # 1K行自定义身份对话
    CustomJSON(filepath=identity_conversations_filepath),  # 重复一次（2个epoch）
    SimpleSpelling(size=200000, split="train"),  # 200K行简单拼写（如"拼写单词'apple'"）
    SpellingBee(size=80000, split="train"),  # 80K行拼写蜜蜂（如"'strawberry'中有几个'r'"）
])  # 总计：460K + 100K + 8K + 1K + 1K + 200K + 80K = ~850K行

val_dataset = TaskMixture([
    SmolTalk(split="test"),  # 24K行测试集
    MMLU(subset="all", split="test", stop=5200),  # 14K行测试集，只使用5.2K以匹配训练比例
    GSM8K(subset="main", split="test", stop=420),  # 1.32K行测试集，只使用420行以匹配训练比例
])  # 总计：24K + 5.2K + 0.42K ~= 30K行

# =============================================================================
# 数据生成器
# =============================================================================
# DataLoader在这里定义，生成 inputs, targets：形状为 (device_batch_size, max_seq_len) 的2D张量
# 一个问题是我们不事先知道最终的 num_iterations。所以我们创建以下全局变量
# 并在数据生成器内更新它们。
last_step = False  # 当到达数据集末尾时将其设为True
approx_progress = 0.0  # 从0到1，表示epoch的进度

def mid_data_generator(split):
    """
    中期训练数据生成器
    
    关键设计：
    1. Token缓冲：累积足够的token以形成完整批次
    2. 分布式处理：每个进程处理不同的文档
    3. 循环遍历：到达数据集末尾时自动重新开始
    4. 进度跟踪：更新全局变量以跟踪训练进度
    5. 内存固定：使用pin_memory加速CPU-GPU传输
    
    参数：
        split: 'train' 或 'val'
        
    生成：
        (inputs, targets) 元组，形状均为 (device_batch_size, max_seq_len)
    """
    global last_step, approx_progress
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    needed_tokens = device_batch_size * max_seq_len + 1  # 形成一个训练批次所需的token数
    token_buffer = deque()  # 使用双端队列作为token缓冲
    # CUDA支持内存固定以加快CPU和GPU之间的传输
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=(device_type == "cuda"))
    cursor = ddp_rank  # 每次递增ddp_world_size，确保每个进程处理不同的文档
    it = 0  # 迭代计数器
    
    while True:
        # 累积足够的token以形成一次迭代
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            token_buffer.extend(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor -= dataset_size  # 循环到数据集开始
                if split == "train":
                    last_step = True  # 设为True以终止训练循环
        
        # 停止条件：如果指定了num_iterations，则遵守它
        it += 1
        if num_iterations > 0 and it >= num_iterations:
            last_step = True  # 设为True以终止训练循环
        
        # 构建inputs/targets并yield
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
        
        # 更新训练进度
        if split == "train":
            if num_iterations > 0:
                approx_progress = it / num_iterations  # 基于最大迭代次数计算进度
            else:
                approx_progress = cursor / dataset_size  # 近似进度为数据集的分数
        yield inputs, targets

train_loader = mid_data_generator("train")
build_val_loader = lambda: mid_data_generator("val")
progress = 0  # 训练进度，从0到1

# =============================================================================
# 学习率调度器
# =============================================================================
def get_lr_multiplier(progress):
    """
    学习率倍数器
    
    策略：
    - 前80%的训练：不衰减（保持1.0）
    - 后20%的训练：线性下降到0
    
    这种策略允许模型在大部分训练时间内充分学习，
    然后在最后阶段平滑地降低学习率以达到更好的收敛。
    """
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

# =============================================================================
# Muon优化器动量调度器
# =============================================================================
def get_muon_momentum(it):
    """
    Muon优化器的动量调度
    
    策略：
    - 从0.85线性增加到0.95（在前300步）
    - 之后保持0.95
    
    较低的初始动量帮助模型更快地适应新数据，
    较高的后期动量提供更稳定的优化。
    """
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# =============================================================================
# 主训练循环
# =============================================================================
x, y = next(train_loader)  # 预取第一批数据
min_val_bpb = float("inf")  # 最小验证集BPB
smooth_train_loss = 0  # 训练损失的EMA（指数移动平均）
ema_beta = 0.9  # EMA衰减因子
total_training_time = 0  # 总训练时间（墙钟时间）
step = 0

while True:
    flops_so_far = num_flops_per_token * total_batch_size * step

    # ============= 同步last_step标志 =============
    # 在分布式设置中同步last_step，避免不同进程不一致导致挂起
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # ============= 定期评估验证集BPB =============
    if eval_every > 0 and (last_step or step % eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"步数 {step:05d} | 验证集 bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # ============= 保存最终检查点 =============
    # 只在主进程上保存，且不是dry_run模式
    if master_process and last_step and not dry_run:
        output_dirname = f"d{depth}"  # 例如 d12
        checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),  # 模型权重
            [opt.state_dict() for opt in optimizers],  # 优化器状态
            {
                "step": step,
                "val_bpb": val_bpb,  # 最后一步的损失
                "model_config": {
                    "sequence_len": max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                },
                "user_config": user_config,  # 训练脚本的输入参数
            }
        )

    if last_step:
        break

    # ============= 单步训练：梯度计算 =============
    synchronize()
    t0 = time.time()
    # 梯度累积循环
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()  # 保存用于日志记录
        loss = loss / grad_accum_steps  # 归一化损失（因为.backward()是梯度求和）
        loss.backward()  # 累积梯度
        x, y = next(train_loader)  # 预取下一批数据（在GPU忙于前向/反向时）
        progress = max(progress, approx_progress)  # 只单调增加进度
    
    # ============= 优化器步进 =============
    # 更新学习率
    lrm = get_lr_multiplier(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    # 更新Muon优化器的动量
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    
    # 执行优化器步进
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)  # 清零梯度
    synchronize()
    t1 = time.time()
    dt = t1 - t0  # 单步训练时间

    # ============= 更新状态 =============
    step += 1

    # ============= 日志记录 =============
    # EMA平滑训练损失
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))  # 去偏EMA
    
    # 计算性能指标
    pct_done = 100 * progress
    tok_per_sec = int(total_batch_size / dt)  # 每秒处理的token数
    flops_per_sec = num_flops_per_token * total_batch_size / dt  # 每秒浮点运算数
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size  # H100 SXM理论峰值FLOPs（bfloat16，无2:4稀疏性）
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # 模型FLOPs利用率（百分比）
    
    if step > 10:
        total_training_time += dt  # 只计算前10步之后的时间
    
    print0(f"步数 {step:05d} ({pct_done:.2f}%) | 损失: {debiased_smooth_loss:.6f} | 学习率倍数: {lrm:.2f} | 耗时: {dt * 1000:.2f}毫秒 | token/秒: {tok_per_sec:,} | MFU: {mfu:.2f} | 总时间: {total_training_time/60:.2f}分钟")
    
    # 每10步记录到WandB
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })

# =============================================================================
# 训练完成：打印最终统计信息
# =============================================================================
print0(f"峰值内存使用: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"总训练时间: {total_training_time/60:.2f}分钟")
print0(f"最低验证集bpb: {min_val_bpb:.4f}")

# =============================================================================
# 记录到实验报告
# =============================================================================
if not dry_run:
    from nanochat.report import get_report
    get_report().log(section="Midtraining", data=[
        user_config,  # 命令行参数
        {  # 训练设置统计
            "Number of iterations": step,
            "DDP world size": ddp_world_size,
        },
        {  # 训练结果统计
            "Minimum validation bpb": min_val_bpb,
        }
    ])

# =============================================================================
# 清理资源
# =============================================================================
wandb_run.finish()  # 结束WandB记录
compute_cleanup()  # 清理计算资源
