"""
强化学习（RL）脚本 - 使用简化的GRPO/REINFORCE在GSM8K数学问题上训练

算法说明：
本脚本实现了一个简化的GRPO（Group Relative Policy Optimization）算法，
实际上更接近经典的REINFORCE算法，但有以下简化：

1. 删除信任区域：无KL散度正则化，不需要参考模型
   - 标准GRPO/PPO会限制策略更新幅度，避免偏离过远
   - 我们删除了这个约束，使算法更简单

2. On-Policy学习：无需PPO的重要性采样比率和裁剪
   - 因为每次都用当前策略采样，不存在策略不匹配问题
   - 不需要PPO的ratio和clip机制

3. GAPO风格的token级归一化：
   - 标准RL通常在序列级别归一化奖励
   - 我们在token级别归一化，每个token有独立的advantage

4. 简化的advantage计算：只使用(r - μ)，不除以σ
   - 标准做法：(r - μ)/σ（z-score归一化）
   - 我们的做法：r - μ（中心化但不缩放）

训练目标：
在GSM8K小学数学问题数据集上，通过强化学习提升模型的数学推理能力。
使用正确答案作为奖励信号（+1或-1），优化模型生成正确解答的概率。

运行方式：

1. 单GPU模式：
   python -m scripts.chat_rl
   说明：适用于调试和小规模实验

2. 多GPU分布式训练（推荐）：
   torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
   说明：使用8个GPU并行训练，显著加速

3. 自定义参数：
   python -m scripts.chat_rl --source=sft --num_samples=32 --temperature=1.2
   说明：从SFT模型开始，每个问题采样32次

技术特性：
- Pass@k评估：从k个样本中选最佳答案
- 批量生成：并行生成多个样本以提高效率
- Token级奖励：每个token根据最终答案正确性获得奖励
- 学习率衰减：线性衰减到0
- 定期评估：跟踪验证集pass@k指标
"""

import os  # 操作系统接口
import itertools  # 迭代器工具
import re  # 正则表达式
import wandb  # Weights & Biases 实验跟踪
import torch  # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb  # 通用工具
from nanochat.checkpoint_manager import save_checkpoint, load_model  # 模型检查点管理
from nanochat.engine import Engine  # 文本生成引擎
from tasks.gsm8k import GSM8K  # GSM8K数学问题数据集

# =============================================================================
# RL 超参数配置
# =============================================================================

# -------------------- 实验跟踪 --------------------
run = "dummy"  # WandB运行名称（"dummy"表示不上传）

# -------------------- 模型和设备 --------------------
source = "sft"  # 模型来源：mid（中期训练）或 sft（监督微调）
dtype = "bfloat16"  # 数据类型

# -------------------- 批次和采样 --------------------
device_batch_size = 8  # 每个设备的批次大小（不会超过此值以避免OOM）
examples_per_step = 16  # 每步的样本数（总计，跨所有进程）注意：是样本数，不是生成数！
num_samples = 16  # 每个样本（问题）的生成数量（用于pass@k）

# -------------------- 生成参数 --------------------
max_new_tokens = 256  # 最大生成token数
temperature = 1.0  # 生成温度
top_k = 50  # Top-k采样参数

# -------------------- 优化器参数 --------------------
unembedding_lr = 0.004  # Unembedding层学习率
embedding_lr = 0.2  # Embedding层学习率
matrix_lr = 0.02  # 矩阵参数学习率（Muon优化器）
weight_decay = 0.0  # 权重衰减
init_lr_frac = 0.05  # 初始学习率占基础学习率的比例

# -------------------- 训练和评估 --------------------
num_epochs = 1  # GSM8K数据集训练epoch数
save_every = 60  # 每隔多少步保存模型
eval_every = 60  # 每隔多少步评估验证集pass@k
eval_examples = 400  # 评估时使用的样本数

# -------------------- 命令行参数覆盖 --------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())  # 允许命令行或配置文件覆盖
user_config = {k: globals()[k] for k in config_keys}  # 保存用户配置用于日志记录
# =============================================================================

# =============================================================================
# 计算环境初始化
# =============================================================================
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0  # 主进程负责日志和检查点保存
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# =============================================================================
# Weights & Biases 日志记录初始化
# =============================================================================
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=run, config=user_config)

# =============================================================================
# 模型和分词器初始化
# =============================================================================
model, tokenizer, meta = load_model(source, device, phase="eval")
engine = Engine(model, tokenizer)  # 用于采样rollouts

# =============================================================================
# 训练和验证数据集
# =============================================================================
train_task = GSM8K(subset="main", split="train")  # GSM8K训练集（约7.5K问题）
val_task = GSM8K(subset="main", split="test")  # GSM8K测试集（约1.3K问题）
num_steps = (len(train_task) // examples_per_step) * num_epochs
print0(f"计算得到的训练步数: {num_steps}")

# =============================================================================
# Rollout生成器 - 生成训练批次
# =============================================================================

@torch.no_grad()
def get_batch():
    """
    批次生成器 - 生成用于RL训练的rollout批次
    
    工作流程：
    1. 获取一个问题，删除参考答案
    2. 生成num_samples个答案（rollouts）
    3. 根据答案正确性计算奖励（+1或-1）
    4. 计算优势函数（rewards - mean）
    5. 返回训练数据（inputs, targets, rewards, advantages）
    
    关键设计：
    - 分布式：每个进程处理不同的问题
    - 批量生成：分批生成以避免OOM
    - Mask处理：只对生成的token计算损失，不对提示计算损失
    - 奖励归一化：减去均值以稳定训练
    """
    assistant_end = tokenizer.encode_special("<|assistant_end|>")  # 填充token（不用于损失计算）
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)  # 每个进程负责不同的样本
    
    for example_idx in itertools.cycle(rank_indices):

        # ============= 1. 获取问题并准备提示 =============
        conversation = train_task[example_idx]

        # 分词对话，删除最后的助手消息，保留<|assistant_start|>以便模型继续补全
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # ============= 2. 生成多个样本（rollouts）=============
        model.eval()  # 确保模型处于评估模式
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size  # 分批生成以防止OOM
        
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF  # 正整数种子
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,  # 必须为每个采样步骤使用不同的种子
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # ============= 3. 计算每个样本的奖励 =============
        rewards = []
        for sample_tokens in generated_token_sequences:
            # 提取生成的token（提示之后的部分）
            generated_tokens = sample_tokens[prefix_length:]
            # 解码生成的回答
            generated_text = tokenizer.decode(generated_tokens)
            # 计算奖励（+1表示正确，-1表示错误）
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # ============= 4. 填充序列使其长度一致 =============
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        
        # 转换为PyTorch tensor
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        
        # 生成自回归的输入和目标
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()  # clone以避免原地修改
        targets[mask_ids[:, 1:] == 0] = -1  # 原地修改。-1是ignore index
        # 注意：Engine对提示token和工具使用token都返回mask=0
        # 所以我们（正确地）不会在提示token或强制token上训练
        
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        
        # ============= 5. 计算优势函数 =============
        # 使用简单的中心化（减去均值）而不是z-score归一化 (x-μ)/σ
        mu = rewards.mean()
        advantages = rewards - mu
        
        # 返回：生成的序列、输入、目标、奖励、优势
        yield generated_token_sequences, inputs, targets, rewards, advantages

# =============================================================================
# GSM8K Pass@k 评估函数
# =============================================================================

def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    评估GSM8K任务并返回评估结果记录
    
    评估方法：Pass@k
    - 对每个问题生成k个答案
    - 如果至少有一个答案正确，则该问题pass
    - pass@k = 通过的问题数 / 总问题数
    
    分布式设置：
    - 所有进程协作处理不同的样本
    - 此函数不进行跨进程的规约（由调用者负责）
    - 逐个yield记录以提供进度反馈
    
    参数：
        task: GSM8K任务对象
        tokenizer: 分词器
        engine: 生成引擎
        max_examples: 最多评估的样本数
        num_samples: 每个问题的生成数量（k）
        max_completion_tokens: 最大补全token数
        temperature: 生成温度
        top_k: Top-k采样参数
        
    生成：
        每个问题的评估记录（包含是否通过）
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        
        # 使用批量生成在Engine内生成k个样本
        assert num_samples <= device_batch_size  # 通常为真，如果不是可以添加循环
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        # 检查每个样本的正确性
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        # 记录结果（为了更复杂的日志记录而设计，虽然目前只记录is_correct）
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# =============================================================================
# 主RL训练循环
# =============================================================================

# ============= 优化器初始化 =============
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,  # 输出层学习率
    embedding_lr=embedding_lr,      # 嵌入层学习率
    matrix_lr=matrix_lr,            # 矩阵参数学习率（Muon优化器）
    weight_decay=weight_decay,      # 权重衰减
)

# 设置初始学习率为基础学习率的一个小比例
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]  # 保存初始学习率以便后续衰减

# ============= 学习率调度器 =============
def get_lr_multiplier(it):
    """
    学习率倍数器：从1.0线性衰减到0.0
    RL阶段使用简单的线性衰减
    """
    lrm = 1.0 - it / num_steps
    return lrm

# ============= 计算每个进程处理的样本数 =============
print0(f"每步总序列数: {examples_per_step * num_samples}")  # 每步总序列数
assert examples_per_step % ddp_world_size == 0, "每步样本数必须能被进程数整除"
examples_per_rank = examples_per_step // ddp_world_size  # 每个GPU处理的样本数
print0(f"每个进程的样本数: {examples_per_rank}")

# ============= 开始训练循环！=============
batch_iterator = get_batch()
for step in range(num_steps):

    # ============= 定期评估模型并记录到WandB =============
    if step % eval_every == 0:
        model.eval()
        # 计算pass@k，k从1到device_batch_size
        passk = torch.zeros(device_batch_size, device=device)
        with autocast_ctx:
            records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=device_batch_size, max_examples=eval_examples, temperature=1.0)
            records = list(records_iter)  # 收集所有记录
        
        # 对每个k值计算pass@k
        for k in range(1, device_batch_size + 1):
            # pass@k: 在k个样本中至少有一个正确
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        
        # 跨进程聚合结果
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item()  # 归一化
        
        # 打印和记录结果
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, device_batch_size + 1)]
        print0(f"步数 {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # ============= 在rollouts上进行前向/反向传播 =============
    # 处理多个样本的rollouts
    rewards_list = []
    sequence_lengths = []
    
    for example_step in range(examples_per_rank):
        # 获取一个批次（对应训练集中的一个样本）
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        
        # 评估损失和梯度
        model.train()  # 确保模型处于训练模式
        
        # 需要额外的循环因为我们不能超过device_batch_size
        assert inputs_all.size(0) % device_batch_size == 0
        num_passes = inputs_all.size(0) // device_batch_size
        
        for pass_idx in range(num_passes):
            # 提取本次pass的批次
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            
            # ============= 计算Policy Gradient损失 =============
            # 计算对数概率。注意：loss计算的是NLL = -logp，所以我们取反
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)  # (B, T)
            
            # 计算PG目标函数：logp * advantage
            # 注意：ignore_index=-1确保无效token的损失为0
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            
            # 归一化：除以有效token数、pass数和examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            
            # 注意：因为是on-policy，不需要PPO的ratio+clip
            
            # 最后，构造要最小化的损失（而不是要最大化的目标）
            loss = -pg_obj
            loss.backward()  # 反向传播累积梯度
            
            print0(f"步数 {step}/{num_steps} | 样本步数 {example_step} | 批次 {pass_idx} | 损失: {loss.item():.6f} | 平均奖励: {rewards.mean().item()}")
        
        # 记录本example的统计信息
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # ============= 记录本步rollouts的统计信息 =============
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    
    # 跨进程聚合
    if ddp:
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    
    print0(f"步数 {step}/{num_steps} | 平均奖励: {mean_reward} | 平均序列长度: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })

    # ============= 更新模型参数 =============
    # 先更新学习率
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    # 然后执行优化器步进
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)  # 清零梯度
    
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # ============= 主进程定期保存模型 =============
    # 跳过第一步，保存最后一步
    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}"  # 基于模型深度命名标签
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
        model_config_kwargs = model.config.__dict__  # 获取模型配置
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),  # 模型权重
            None,  # 注意：RL阶段不保存优化器状态（节省空间）
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ 已保存模型检查点到 {checkpoint_dir}")

# =============================================================================
# 训练完成：记录到实验报告
# =============================================================================
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[
    user_config,  # 命令行参数和配置
])

# =============================================================================
# 清理资源
# =============================================================================
wandb_run.finish()  # 结束WandB记录
compute_cleanup()  # 清理分布式环境
