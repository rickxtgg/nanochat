"""
监督微调（SFT）脚本 - 将基础模型转换为对话模型

功能说明：
本脚本实现监督微调（Supervised Fine-Tuning, SFT）训练流程，
将预训练的基础模型（base model）或中期训练模型（midtrained model）
微调为能够进行自然对话的对话模型。

训练数据：
SFT阶段使用多种任务的混合数据进行训练，包括：
1. ARC-Easy/Challenge: AI推理挑战（多选题）
2. GSM8K: 小学数学问题（需要推理和计算）
3. SmolTalk: 通用对话数据（日常交流）
4. CustomJSON: 自定义合成对话（模型身份标识）
5. SimpleSpelling/SpellingBee: 拼写相关任务

运行方式：

1. 单GPU调试模式：
   python -m scripts.chat_sft
   说明：适用于调试和小规模实验

2. 多GPU分布式训练（推荐）：
   torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
   说明：使用8个GPU并行训练，显著加速训练过程

3. 自定义参数：
   python -m scripts.chat_sft --source=mid --num_epochs=2 --device_batch_size=8
   说明：从中期训练模型开始，训练2个epoch

4. 使用WandB跟踪：
   python -m scripts.chat_sft --run=my_sft_run
   说明：启用 Weights & Biases 实验跟踪

技术特性：
- 混合任务训练：同时学习推理、对话、知识问答等多种能力
- 梯度累积：支持大批次训练即使GPU内存有限
- 学习率调度：线性衰减到0
- 混合优化器：Muon用于矩阵参数，AdamW用于embedding
- 定期评估：验证集损失和多选题准确率
- 动态批处理：自动处理不同长度的对话序列
- Mask机制：只对助手回复计算损失，不对用户输入计算损失

输出：
- 训练好的对话模型检查点
- 验证集损失曲线
- MMLU和ARC-Easy准确率
- 实验报告记录
"""

import os
# 设置 PyTorch CUDA 内存分配策略：使用可扩展段，减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb  # Weights & Biases 实验跟踪
import torch  # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练
from contextlib import nullcontext  # 空上下文管理器

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type  # 通用工具
from nanochat.checkpoint_manager import load_model, save_checkpoint  # 模型检查点管理
from nanochat.engine import Engine  # 文本生成引擎
from scripts.chat_eval import run_chat_eval  # 评估函数

# 导入任务数据集
from tasks.common import TaskMixture  # 任务混合工具
from tasks.arc import ARC  # AI推理挑战
from tasks.gsm8k import GSM8K  # 小学数学问题
from tasks.smoltalk import SmolTalk  # 通用对话数据
from tasks.customjson import CustomJSON  # 自定义JSON对话
from tasks.spellingbee import SimpleSpelling, SpellingBee  # 拼写任务

# =============================================================================
# SFT 超参数配置
# =============================================================================

# -------------------- 实验跟踪 --------------------
run = "dummy"  # WandB运行名称（"dummy"表示不上传到WandB）

# -------------------- 输入模型选项 --------------------
source = "mid"  # 模型来源：base（基础模型）或 mid（中期训练模型）
model_tag = None  # 可选：模型标签
step = None  # 可选：特定训练步数的检查点

# -------------------- 计算和精度 --------------------
device_type = ""  # 设备类型：cuda|cpu|mps（空值表示自动检测）
dtype = "bfloat16"  # 数据类型：bfloat16或float32
device_batch_size = 4  # 每个设备的批次大小（设置为不会OOM的最大值）

# -------------------- 优化参数 --------------------
num_epochs = 1  # 训练epoch数
num_iterations = -1  # 覆盖迭代次数（-1表示禁用，使用num_epochs推导）
target_examples_per_step = 32  # 每步目标样本数（通过梯度累积实现）
unembedding_lr = 0.004  # Unembedding（输出层）学习率
embedding_lr = 0.2  # Embedding 学习率
matrix_lr = 0.02  # 矩阵参数学习率（Muon优化器）
weight_decay = 0.0  # 权重衰减（L2正则化）
init_lr_frac = 0.02  # 初始学习率占基础学习率的比例

# -------------------- 评估和日志 --------------------
eval_every = 100  # 每隔多少步评估验证集损失
eval_steps = 100  # 验证集评估步数
eval_metrics_every = 200  # 每隔多少步评估准确率指标
eval_metrics_max_problems = 1024  # 评估指标时最多使用的问题数

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
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# =============================================================================
# Weights & Biases 日志记录初始化
# =============================================================================
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=run, config=user_config, save_code=True)

# =============================================================================
# 加载模型和分词器
# =============================================================================
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
orig_model = model  # 保留原始模型（未编译）
# model = torch.compile(model, dynamic=True)  # 由于输入长度可变，torch.compile效果不佳，故注释
engine = Engine(model, tokenizer)  # 用于内联模型评估

# =============================================================================
# SFT训练任务数据混合
# =============================================================================
# 加载多种任务的混合数据，涵盖推理、对话、拼写等多个领域
identity_conversations_filepath = os.path.join(get_base_dir(), "identity_conversations.jsonl")
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"),  # 2.3K 行 - AI推理挑战（简单）
    ARC(subset="ARC-Challenge", split="train"),  # 1.1K 行 - AI推理挑战（困难）
    GSM8K(subset="main", split="train"),  # 8K 行 - 小学数学问题
    SmolTalk(split="train", stop=10_000),  # 10K 行 - 通用对话数据
    CustomJSON(filepath=identity_conversations_filepath),  # 1K 行 - 自定义身份对话
    SimpleSpelling(size=300, split="train"),  # 300 行 - 简单拼写（如"拼写单词'apple'"）
    SpellingBee(size=300, split="train"),  # 300 行 - 拼写蜜蜂（如"'strawberry'中有几个'r'"）
])  # 总计：2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K ≈ 23K 行
val_ds = SmolTalk(split="test")  # 验证集：通用对话，24K行（实际不会全部使用）

# =============================================================================
# SFT数据加载器
# =============================================================================

def sft_data_generator(dataset, batch_size):
    """
    SFT数据生成器 - 将对话数据转换为训练批次
    
    关键设计：
    1. 动态填充：不同对话长度不同，需要填充到统一长度
    2. Loss Mask：只对助手回复计算损失，用户输入和填充位置的损失被忽略
    3. 分布式处理：每个进程处理不同的数据子集
    4. 无限循环：持续生成批次直到训练结束
    """
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")  # 使用<|assistant_end|>作为填充token（loss中会被mask）
    
    def collate_and_yield(batch):
        """
        将一批tokenized对话整理成统一格式的tensor
        
        处理步骤：
        1. 确定批次最大长度（需要-1因为seq[n]产生inputs[n-1]和targets[n-1]）
        2. 创建填充后的inputs和targets tensor
        3. 根据mask将不需要计算损失的位置设为-1（ignore index）
        """
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1  # 序列长度n产生n-1个输入/目标对
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)  # -1是PyTorch的ignore index
        
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]  # 输入是除最后一个token外的所有token
            
            # 应用mask：只对助手回复计算损失
            row_targets = ids_tensor[1:]  # 目标是除第一个token外的所有token
            # mask[1:] 忽略BOS token的mask（BOS永远不是目标）
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1  # mask为0的位置设为-1（忽略）
            targets[i, :n-1] = row_targets
        
        inputs = inputs.to(device)  # 移动到GPU
        targets = targets.to(device)
        return inputs, targets
    
    # 无限循环遍历数据集，每个epoch结束后自动重新开始
    batch = []
    while True:
        # 分布式处理：每个进程处理不同的样本
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)  # 将对话转换为token序列和mask
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

# =============================================================================
# 梯度累积计算
# =============================================================================
examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, "目标每步样本数必须能被实际每步样本数整除"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

# =============================================================================
# 训练迭代次数计算
# =============================================================================
if num_iterations == -1:
    # 根据num_epochs和数据集大小推导迭代次数
    assert num_epochs > 0, "当num_iterations为-1时，num_epochs必须大于0"
    num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
    
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

# =============================================================================
# 优化器初始化
# =============================================================================
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,  # 输出层学习率
    embedding_lr=embedding_lr,      # 嵌入层学习率
    matrix_lr=matrix_lr,            # 矩阵参数学习率（Muon优化器）
    weight_decay=weight_decay,      # 权重衰减
)

# 设置初始学习率为基础学习率的一个小比例（warmup）
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]  # 保存初始学习率以便后续衰减

# =============================================================================
# 主训练循环
# =============================================================================

# 学习率调度器：线性衰减到0
def get_lr_multiplier(it):
    """
    学习率倍数器：从1.0线性衰减到0.0
    SFT阶段不使用warmup，直接从初始学习率开始衰减
    """
    lrm = 1.0 - it / num_iterations
    return lrm

# 开始训练！
step = 0
train_iter = iter(train_loader)
for step in range(num_iterations):
    last_step = step == num_iterations - 1  # 是否为最后一步

    # ============= 定期评估验证集损失 =============
    if last_step or step % eval_every == 0:
        model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean()  # 对多个批次取平均
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)  # 多卡平均
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        model.train()

    # ============= 定期评估多选题准确率 =============
    # 多选题评估速度快，可以频繁运行
    if last_step or (step > 0 and step % eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            # 注意：因为在no_grad下，可以使用2倍批次大小以加速评估
            metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            "step": step,
            **metrics,
        })
        model.train()

    if last_step:
        break

    # ============= 单步训练：梯度计算和累积 =============
    num_tokens = torch.tensor(0, device=device)  # 记录实际参与训练的token数（不包括被mask的）
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach()  # 保存用于日志记录
        loss = loss / grad_accum_steps  # 归一化损失（因为.backward()是梯度求和）
        loss.backward()  # 累积梯度
        num_tokens += (train_targets >= 0).sum()  # 统计有效token数（targets >= 0）
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)  # 多卡求和

    # ============= 学习率调度 =============
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm  # 应用线性衰减

    # ============= 优化器步进 =============
    for opt in optimizers:
        opt.step()  # 更新参数
    model.zero_grad(set_to_none=True)  # 清零梯度

    # ============= 日志记录 =============
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    })
    step += 1

# =============================================================================
# 训练完成：保存模型检查点
# =============================================================================
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag = f"d{depth}"  # 基于基础模型深度命名模型标签
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
    model_config_kwargs = model.config.__dict__  # 获取模型配置（利用GPTConfig的简单性）
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),  # 模型权重
        None,  # 注意：SFT阶段不保存优化器状态（节省空间）
        {
            "step": step,
            "val_loss": val_loss,
            **metrics,  # 包含最后一步的评估指标
            "model_config": model_config_kwargs,
        }
    )
    print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# =============================================================================
# 记录到实验报告
# =============================================================================
from nanochat.report import get_report
get_report().log(section="Chat SFT", data=[
    user_config,  # 命令行参数
    {
        "Training rows": len(train_ds),  # 训练样本数
        "Number of iterations": num_iterations,  # 迭代次数
        "Training loss": train_loss_item,  # 最终训练损失
        "Validation loss": val_loss,  # 最终验证损失
    },
])

# =============================================================================
# 清理资源
# =============================================================================
wandb_run.finish()  # 结束WandB记录
compute_cleanup()  # 清理分布式环境
