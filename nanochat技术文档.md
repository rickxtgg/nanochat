# nanochat 项目技术文档

> **版本**: v0.1.0  
> **最后更新**: 2025年11月  
> **文档类型**: 保姆级技术文档

---

## 📋 目录

- [一、项目概述](#一项目概述)
- [二、核心架构详解](#二核心架构详解)
- [三、模块功能详解](#三模块功能详解)
- [四、训练流程详解](#四训练流程详解)
- [五、使用指南](#五使用指南)
- [六、总结与最佳实践](#六总结与最佳实践)

> 💡 **补充阅读**：
> - [训练参数计算详解](./训练参数计算详解.md) - 深入解析各个训练参数如何计算和相互影响
> - [模型规模与中文语料适配指南](./模型规模与中文语料适配指南.md) - d4-d32 各规模参数详解与中文/中英文训练策略

---

## 一、项目概述

### 1.1 项目简介

**nanochat** 是一个完整的端到端 LLM（大语言模型）实现项目，旨在用 **100 美元**的预算训练出一个类似 ChatGPT 的聊天模型。这是一个精简、清晰、可破解的代码库，设计用于在单个 8XH100 GPU 节点上运行完整的训练流程。

**核心特点**：
- 🎯 **全栈实现**：涵盖分词、预训练、微调、评估、推理和 Web 服务
- 💰 **成本可控**：$100（4小时）到 $1000（约42小时）的不同规模训练方案
- 🔧 **极简设计**：约 8,300 行代码，45 个文件，依赖最小化
- 📚 **教学导向**：作为 Eureka Labs 的 LLM101n 课程顶点项目
- 🚀 **高性能**：使用 Muon 优化器和 PyTorch 编译加速

**技术亮点**：
- GPT 架构实现（含 RoPE、QK Norm、ReLU²、MQA/GQA）
- 自定义 Rust BPE 分词器（高效训练和推理）
- 分布式训练支持（DDP）
- KV Cache 推理引擎
- 工具调用能力（Python REPL）
- 强化学习支持（GRPO 算法）

### 1.2 项目结构总览

```
nanochat/
├── nanochat/              # 核心库代码
│   ├── gpt.py            # GPT 模型定义
│   ├── engine.py         # 推理引擎（含 KV Cache）
│   ├── dataloader.py     # 分布式数据加载器
│   ├── tokenizer.py      # BPE 分词器封装
│   ├── adamw.py          # 分布式 AdamW 优化器
│   ├── muon.py           # Muon 优化器
│   ├── checkpoint_manager.py  # 检查点管理
│   ├── configurator.py   # 配置管理系统
│   ├── common.py         # 通用工具函数
│   ├── dataset.py        # 预训练数据下载工具
│   ├── core_eval.py      # CORE 评估指标
│   ├── loss_eval.py      # 损失评估（bits per byte）
│   ├── execution.py      # 工具执行（Python REPL）
│   ├── report.py         # 报告生成工具
│   └── ui.html           # Web UI 界面
├── scripts/              # 可执行脚本
│   ├── tok_train.py      # 分词器训练
│   ├── tok_eval.py       # 分词器评估
│   ├── base_train.py     # 基础模型预训练
│   ├── base_eval.py      # 基础模型评估
│   ├── base_loss.py      # 基础模型损失计算
│   ├── mid_train.py      # 中间训练（对话格式适应）
│   ├── chat_sft.py       # 监督微调
│   ├── chat_rl.py        # 强化学习
│   ├── chat_eval.py      # 聊天模型评估
│   ├── chat_cli.py       # 命令行聊天界面
│   └── chat_web.py       # Web 聊天界面
├── tasks/                # 评估任务定义
│   ├── common.py         # 任务基类
│   ├── arc.py            # ARC 科学问答
│   ├── gsm8k.py          # 数学问题
│   ├── humaneval.py      # 代码评估
│   ├── mmlu.py           # 多领域选择题
│   ├── smoltalk.py       # 对话数据集
│   ├── spellingbee.py    # 拼写任务
│   └── customjson.py     # 自定义 JSON 任务
├── rustbpe/              # Rust 实现的 BPE 分词器
│   ├── Cargo.toml        # Rust 项目配置
│   └── src/lib.rs        # Rust 源码
├── tests/                # 测试文件
├── dev/                  # 开发工具
├── speedrun.sh           # 快速训练脚本（$100 预算）
├── run1000.sh            # 完整训练脚本（$1000 预算）
└── pyproject.toml        # Python 项目配置
```

### 1.3 技术栈

**编程语言**：
- Python 3.10+（主要实现）
- Rust（高性能分词器）
- HTML/CSS/JavaScript（Web UI）

**核心依赖**：
- PyTorch 2.8+（深度学习框架）
- tiktoken（分词推理）
- tokenizers（HuggingFace 分词器）
- fastapi + uvicorn（Web 服务）
- datasets（数据集加载）
- wandb（实验跟踪，可选）

**构建工具**：
- uv（Python 包管理）
- maturin（Rust-Python 绑定）
- cargo（Rust 构建工具）

---

## 二、核心架构详解

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     nanochat 完整训练流程                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │    阶段 1: 数据准备与分词器训练           │
        │  - 下载预训练数据（FineWeb-Edu）         │
        │  - 训练 BPE 分词器（vocab_size=65536）   │
        │  - 评估压缩率                            │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │    阶段 2: 基础模型预训练 (Base)          │
        │  - 在原始文本上训练（20B tokens）         │
        │  - 使用 Muon + AdamW 优化器              │
        │  - 评估 CORE 指标和困惑度                │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │    阶段 3: 中间训练 (Midtraining)         │
        │  - 学习对话格式和特殊 token               │
        │  - 引入工具使用能力                       │
        │  - 混合多选题和对话数据                   │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │    阶段 4: 监督微调 (SFT)                 │
        │  - 在高质量对话上精调                     │
        │  - 任务混合：ARC/GSM8K/SmolTalk          │
        │  - 注入身份个性                          │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │    阶段 5: 强化学习 (RL, 可选)             │
        │  - GRPO 算法优化                         │
        │  - 主要针对 GSM8K 数学问题               │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │    阶段 6: 部署与推理                     │
        │  - CLI 聊天界面                          │
        │  - Web UI 服务                           │
        │  - KV Cache 加速推理                     │
        └─────────────────────────────────────────┘
```

### 2.2 模型架构设计

**GPT 模型特性**（`nanochat/gpt.py`）：

1. **位置编码**：使用 RoPE（Rotary Position Embeddings）而非传统的位置嵌入
2. **注意力机制**：支持 MQA（Multi-Query Attention）和 GQA（Grouped-Query Attention）
3. **归一化**：使用无参数的 RMSNorm
4. **激活函数**：MLP 使用 ReLU² 激活
5. **稳定性技术**：
   - QK Norm（查询和键的归一化）
   - Logits Softcap（logits 裁剪到 [-15, 15]）
6. **解耦权重**：Token 嵌入和 LM Head 不共享权重

**模型规模配置**：

```python
# depth=20 (d20) 模型，约 561M 参数
depth = 20
model_dim = depth * 64 = 1280
num_heads = (model_dim + 127) // 128 = 10
head_dim = 128
vocab_size = 65536
sequence_len = 2048

# 参数计算
# - 嵌入层：vocab_size × model_dim
# - Transformer 层：depth × (注意力 + MLP)
# - LM Head：model_dim × vocab_size
```

### 2.3 优化器设计

**混合优化器策略**（`nanochat/muon.py` + `nanochat/adamw.py`）：

```python
# 1. Muon 优化器 - 用于线性层权重矩阵
#    - 基于牛顿方法的二阶优化器
#    - 学习率：0.02
#    - 动量：0.85 -> 0.95（逐步增加）

# 2. AdamW 优化器 - 用于嵌入层和 LM Head
#    - 嵌入层学习率：0.2（缩放 ∝1/√d_model）
#    - LM Head 学习率：0.004
#    - Betas：(0.8, 0.95)
#    - 权重衰减：0.0
```

**学习率调度**：
- 预热（Warmup）：0% 的训练步数
- 恒定期：80% 的训练步数
- 衰减期（Warmdown）：20% 的训练步数，线性衰减到 0

### 2.4 数据处理流程

**预训练数据**（`nanochat/dataset.py` + `nanochat/dataloader.py`）：

```
FineWeb-Edu (来源)
    │
    ├─→ 下载 Parquet 分片（~1822 个分片，每个 250M 字符）
    │
    ├─→ 流式读取和分词
    │   - 使用 deque 缓冲区
    │   - 批量分词（batch_size=128）
    │   - 多线程加速（4 线程）
    │
    └─→ 构造训练批次
        - 序列长度：2048 tokens
        - 批次大小：524,288 tokens（总计）
        - 梯度累积：自动计算
```

**对话数据格式**（`nanochat/tokenizer.py::render_conversation`）：

```
<|bos|>
<|user_start|>用户消息内容<|user_end|>
<|assistant_start|>助手回复内容<|assistant_end|>
<|user_start|>继续对话...<|user_end|>
<|assistant_start|>继续回复...<|assistant_end|>
```

**工具调用格式**：

```
<|assistant_start|>
我来帮你计算：
<|python_start|>2 + 2<|python_end|>
<|output_start|>4<|output_end|>
结果是 4。
<|assistant_end|>
```

---

## 三、模块功能详解

### 3.1 核心模块

#### 3.1.1 GPT 模型（`nanochat/gpt.py`）

**类结构**：

```python
GPTConfig:
    - sequence_len: int = 2048      # 最大序列长度
    - vocab_size: int = 65536       # 词汇表大小
    - n_layer: int = 20             # Transformer 层数
    - n_head: int = 10              # 注意力头数
    - n_kv_head: int = 10           # KV 头数（MQA/GQA）
    - n_embd: int = 1280            # 模型维度

GPT(nn.Module):
    - transformer:
        - wte: Embedding            # Token 嵌入
        - h: ModuleList[Block]      # Transformer 块
    - lm_head: Linear               # 输出投影
    - cos, sin: Tensor              # RoPE 嵌入（缓存）
```

**关键方法**：

1. **`forward(idx, targets=None, kv_cache=None)`**：
   - 训练时：返回交叉熵损失
   - 推理时：返回 logits
   - 支持 KV Cache 加速

2. **`generate(tokens, max_tokens, temperature, top_k)`**：
   - 朴素的自回归生成
   - 适用于简单推理

3. **`setup_optimizers(...)`**：
   - 自动配置混合优化器
   - 根据模型维度缩放学习率

4. **`estimate_flops()`**：
   - 估算每个 token 的 FLOPs
   - 用于计算训练效率（MFU）

**注意力机制**（`CausalSelfAttention`）：

```python
# 标准流程
Q = norm(RoPE(query_proj(x)))
K = norm(RoPE(key_proj(x)))
V = value_proj(x)

# KV Cache 支持
if kv_cache:
    K, V = kv_cache.insert_kv(layer_idx, K, V)

# Scaled Dot-Product Attention
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

**MLP 模块**：

```python
def forward(x):
    x = linear1(x)           # [B, T, D] -> [B, T, 4D]
    x = relu(x).square()     # ReLU² 激活
    x = linear2(x)           # [B, T, 4D] -> [B, T, D]
    return x
```

#### 3.1.2 推理引擎（`nanochat/engine.py`）

**KV Cache 实现**：

```python
class KVCache:
    # 形状：(n_layers, 2, batch_size, n_heads, seq_len, head_dim)
    # - 2: K 和 V
    # - 动态增长：按需扩展 seq_len 维度（1024 步长）
    
    def insert_kv(layer_idx, k, v):
        # 插入新的 K/V 到缓存
        # 返回完整的历史 K/V（作为视图）
        # 自动更新 pos 指针
```

**Engine 生成流程**：

```python
class Engine:
    def generate(tokens, num_samples, max_tokens, temperature, top_k):
        # 1. 预填充阶段（Prefill）
        #    - 单批次处理提示词
        #    - 初始化 KV Cache
        
        # 2. 复制 KV Cache
        #    - 为每个样本创建独立的缓存副本
        
        # 3. 逐 token 生成（Decode）
        #    - 每步只前向传播 1 个 token
        #    - 利用 KV Cache 避免重复计算
        #    - 支持批量采样（num_samples > 1）
        
        # 4. 工具调用状态机
        #    - 检测 <|python_start|> token
        #    - 执行 Python 表达式
        #    - 强制注入 <|output_start|>结果<|output_end|>
```

**工具使用（Calculator）**：

```python
def use_calculator(expr):
    # 安全执行数学表达式或字符串操作
    # 支持：
    #   - 数学运算：1 + 2 * 3
    #   - 字符串方法："strawberry".count("r")
    # 
    # 安全措施：
    #   - 3 秒超时
    #   - 禁止危险操作（import, exec, __等）
    #   - 空命名空间
```

#### 3.1.3 分词器（`nanochat/tokenizer.py`）

**双实现架构**：

1. **HuggingFaceTokenizer**：
   - 用于训练和推理
   - 灵活但稍慢

2. **RustBPETokenizer**（推荐）：
   - 训练：使用 `rustbpe`（Rust 实现）
   - 推理：使用 `tiktoken`（C++ 实现）
   - 性能优异

**训练流程**：

```python
# scripts/tok_train.py

# 1. 下载数据
dataset.download_shards(n=8)  # ~2B 字符

# 2. 训练分词器
tokenizer = RustBPETokenizer.train_from_iterator(
    text_iterator=text_stream,
    vocab_size=65536,
)

# 3. 保存
tokenizer.save("tokenizer/")
# - tokenizer.pkl (tiktoken Encoding 对象)
# - token_bytes.pt (每个 token 的字节长度)
```

**特殊 Token**：

```python
SPECIAL_TOKENS = [
    "<|bos|>",              # 文档开始
    "<|user_start|>",       # 用户消息开始
    "<|user_end|>",         # 用户消息结束
    "<|assistant_start|>",  # 助手消息开始
    "<|assistant_end|>",    # 助手消息结束
    "<|python_start|>",     # Python 工具调用开始
    "<|python_end|>",       # Python 工具调用结束
    "<|output_start|>",     # 工具输出开始
    "<|output_end|>",       # 工具输出结束
]
```

**对话渲染**：

```python
def render_conversation(conversation, max_tokens=2048):
    # 输入：{"messages": [{"role": "user", "content": "..."}, ...]}
    # 输出：
    #   - ids: List[int] - token 序列
    #   - mask: List[int] - 监督掩码（1=训练，0=不训练）
    
    # 规则：
    # - 用户消息：mask=0（不训练）
    # - 助手消息：mask=1（训练）
    # - 工具输出：mask=0（测试时由 Python 生成）
```

#### 3.1.4 数据加载器（`nanochat/dataloader.py`）

**分布式流式加载**：

```python
def tokenizing_distributed_data_loader(B, T, split, device):
    # B: batch size（每设备）
    # T: sequence length
    # split: "train" 或 "val"
    
    # 流程：
    # 1. 从 Parquet 文件流式读取文档
    #    - 各 rank 读取不同的分片（rank, rank+world_size, ...）
    #    
    # 2. 批量分词
    #    - tokenizer_batch_size=128
    #    - num_threads=4
    #    
    # 3. 累积到 deque 缓冲区
    #    - 需要 B*T+1 个 token 才 yield
    #    
    # 4. 构造 (inputs, targets)
    #    - inputs: tokens[:-1]
    #    - targets: tokens[1:]
    #    - 形状：(B, T)
    
    # 特性：
    # - 无限循环（无限 epoch）
    # - 自动处理分布式
    # - 内存高效（流式）
```

### 3.2 训练脚本

#### 3.2.1 基础模型训练（`scripts/base_train.py`）

**主要流程**：

```python
# 1. 配置解析
#    - 使用 configurator.py 从 CLI 读取参数
#    - 支持配置文件覆盖

# 2. 计算初始化
#    - DDP 设置（如果多 GPU）
#    - 设备选择（CUDA/MPS/CPU）
#    - 随机种子固定（reproducibility）

# 3. 模型创建
#    - depth -> (n_layer, n_embd, n_head)
#    - 在 meta device 上初始化（节省内存）
#    - 转移到设备并初始化权重

# 4. 优化器配置
#    - Muon for Linear 层
#    - AdamW for Embedding + LM Head
#    - 学习率根据 d_model 缩放

# 5. 数据加载器
#    - Train: 无限流
#    - Val: 按需构建

# 6. 训练循环
for step in range(num_iterations + 1):
    # 评估验证损失
    if step % eval_every == 0:
        val_bpb = evaluate_bpb(model, val_loader, ...)
    
    # 评估 CORE 指标
    if step % core_metric_every == 0:
        core_score = evaluate_model(model, tokenizer, ...)
    
    # 采样展示
    if step % sample_every == 0:
        generate_samples(...)
    
    # 训练步骤
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        loss.backward()
        x, y = next(train_loader)  # 预取
    
    # 梯度裁剪 + 优化器步进
    clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    # 保存检查点
    if last_step:
        save_checkpoint(...)

# 7. 报告生成
get_report().log(section="Base model training", data=...)
```

**关键超参数**：

```python
# 模型架构
depth = 20                    # 模型深度
max_seq_len = 2048           # 序列长度

# 训练规模
target_param_data_ratio = 20  # Chinchilla 比例
total_batch_size = 524288     # ~0.5M tokens/step
device_batch_size = 32        # 每设备批次大小

# 优化器
matrix_lr = 0.02             # Muon 学习率
embedding_lr = 0.2           # 嵌入学习率
unembedding_lr = 0.004       # LM Head 学习率
grad_clip = 1.0              # 梯度裁剪

# 学习率调度
warmup_ratio = 0.0           # 预热比例
warmdown_ratio = 0.2         # 衰减比例
final_lr_frac = 0.0          # 最终学习率比例

# 评估频率
eval_every = 250             # 验证评估
core_metric_every = 2000     # CORE 评估
sample_every = 2000          # 采样展示
```

#### 3.2.2 监督微调（`scripts/chat_sft.py`）

**数据混合**：

```python
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"),        # 2.3K 科学问答
    ARC(subset="ARC-Challenge", split="train"),   # 1.1K 挑战问答
    GSM8K(subset="main", split="train"),          # 8K 数学问题
    SmolTalk(split="train", stop=10_000),         # 10K 对话
    CustomJSON(filepath="identity_conversations.jsonl"),  # 1K 身份对话
    SimpleSpelling(size=300, split="train"),      # 300 拼写任务
    SpellingBee(size=300, split="train"),         # 300 字母计数
])
# 总计：~23K 训练样本
```

**数据处理**：

```python
def sft_data_generator(dataset, batch_size):
    # 1. 迭代数据集（分布式：每个 rank 处理不同样本）
    for i in range(ddp_rank, len(dataset), ddp_world_size):
        doc = dataset[i]
        
        # 2. 渲染对话为 token 序列
        ids, mask = tokenizer.render_conversation(doc)
        batch.append((ids, mask))
        
        # 3. 批次对齐（padding）
        if len(batch) == batch_size:
            # - 找最长序列
            # - 用 <|assistant_end|> 填充
            # - mask=0 的位置 target=-1（ignore_index）
            yield collate_and_yield(batch)
```

**训练特点**：

- **源模型**：可选 `base` 或 `mid`（中间训练后的模型）
- **优化器**：同样使用 Muon + AdamW，但学习率降低（init_lr_frac=0.02）
- **学习率调度**：线性衰减到 0
- **评估**：验证损失 + MMLU/ARC 准确率
- **epoch 数**：通常 1 epoch 足够

#### 3.2.3 强化学习（`scripts/chat_rl.py`）

**GRPO 算法**（Group Relative Policy Optimization）：

```python
# 1. 采样阶段
for problem in dataset:
    prompt_tokens = tokenizer.render_for_completion(problem)
    
    # 生成多个候选（num_samples=4）
    completions, masks = engine.generate_batch(
        prompt_tokens,
        num_samples=4,
        temperature=1.0,
    )
    
    # 评估奖励
    for completion in completions:
        reward = task.evaluate(problem, completion)
        rewards.append(reward)

# 2. 优势计算
mean_reward = mean(rewards)
advantages = [r - mean_reward for r in rewards]

# 3. 策略优化
for completion, advantage in zip(completions, advantages):
    # 计算对数概率比
    log_probs_new = model.forward_log_probs(completion)
    log_probs_old = log_probs_new.detach()  # 参考策略
    
    # GRPO 损失
    ratio = exp(log_probs_new - log_probs_old)
    loss = -advantage * ratio
    loss.backward()

optimizer.step()
```

**特性**：
- **无需奖励模型**：直接使用任务评估函数
- **主要用于 GSM8K**：数学问题有明确的对错
- **On-policy**：每批次重新采样
- **Group normalization**：优势在组内归一化

### 3.3 评估系统

#### 3.3.1 CORE 评估（`nanochat/core_eval.py`）

**CORE 指标**（来自 DCLM 论文）：

```python
# 定义
CORE = centered_mean([
    ARC-Challenge,
    ARC-Easy,
    HellaSwag,
    MMLU,
    OpenBookQA,
    PIQA,
    Winogrande,
])

# centered_mean: 将每个任务分数居中到 [0, 1]，然后平均
def centered_mean(scores):
    centered = [(s - random_baseline) / (1 - random_baseline) 
                for s in scores]
    return mean(centered)
```

**评估流程**：

```python
def evaluate_model(model, tokenizer, device, max_per_task=500):
    # 1. 加载所有 CORE 任务
    tasks = {
        "ARC-Challenge": ARC("ARC-Challenge", "test"),
        "ARC-Easy": ARC("ARC-Easy", "test"),
        # ... 其他任务
    }
    
    # 2. 对每个任务评估
    for task_name, task in tasks.items():
        correct = 0
        total = min(max_per_task, len(task))
        
        for problem in task[:total]:
            # 渲染为多选题
            prompt = render_mc(problem.question, letters, choices)
            
            # 计算每个选项的困惑度
            perplexities = []
            for choice in choices:
                tokens = tokenizer.encode(prompt + choice)
                loss = model(tokens, targets)
                perplexities.append(exp(loss))
            
            # 选择困惑度最低的
            prediction = argmin(perplexities)
            if prediction == problem.answer:
                correct += 1
        
        accuracy = correct / total
        scores[task_name] = accuracy
    
    # 3. 计算 CORE 分数
    core_score = centered_mean(scores.values())
    return core_score, scores
```

#### 3.3.2 聊天评估（`scripts/chat_eval.py`）

**支持的任务**：

1. **MMLU**：多领域选择题（57 个子集）
2. **ARC-Easy/Challenge**：科学推理
3. **GSM8K**：小学数学应用题
4. **HumanEval**：Python 代码生成
5. **ChatCORE**：对话版 CORE 评估

**评估模式**：

```python
# 分类任务（Multiple Choice）
def evaluate_categorical(task, model, tokenizer, engine):
    for problem in task:
        # 1. 渲染提示
        prompt = render_mc(problem.question, choices)
        tokens = tokenizer.encode(prompt)
        
        # 2. 生成回复
        completion = engine.generate_batch(tokens, temperature=0)
        answer = tokenizer.decode(completion)
        
        # 3. 提取字母答案
        predicted_letter = extract_letter(answer)
        
        # 4. 评估
        correct = (predicted_letter == problem.answer)

# 生成任务（Generative）
def evaluate_generative(task, model, tokenizer, engine):
    for problem in task:
        # 1. 渲染提示
        prompt = problem.question
        tokens = tokenizer.encode(prompt)
        
        # 2. 生成回复
        completion = engine.generate_batch(
            tokens,
            max_tokens=512,
            temperature=0,
        )
        answer = tokenizer.decode(completion)
        
        # 3. 任务特定评估
        correct = task.evaluate(problem, answer)
        # 例如：GSM8K 提取最终数字并比较
```

### 3.4 工具和辅助模块

#### 3.4.1 检查点管理（`nanochat/checkpoint_manager.py`）

```python
def save_checkpoint(checkpoint_dir, step, model_state, optimizer_states, meta):
    # 保存：
    # - model_state.pt: 模型权重
    # - optimizer_0.pt, optimizer_1.pt: 优化器状态
    # - meta.pt: 元数据（step, 配置等）
    pass

def load_model(source, device, phase="eval", model_tag=None, step=None):
    # source: "base", "mid", "sft", "rl"
    # phase: "train" (加载优化器) 或 "eval" (仅模型)
    # 
    # 自动查找：
    # - base_checkpoints/{model_tag}/
    # - chatmid_checkpoints/{model_tag}/
    # - chatsft_checkpoints/{model_tag}/
    # - chatrl_checkpoints/{model_tag}/
    
    return model, tokenizer, meta
```

#### 3.4.2 配置器（`nanochat/configurator.py`）

**简易配置系统**：

```python
# 用法 1：命令行参数
python script.py --depth=20 --device_batch_size=16

# 用法 2：配置文件
# config/my_config.py
depth = 26
device_batch_size = 16

python script.py config/my_config.py

# 用法 3：混合
python script.py config/my_config.py --depth=32

# 实现原理：
# 1. 扫描 sys.argv
# 2. 执行配置文件（exec(open(file).read())）
# 3. 解析 --key=value 并更新 globals()
```

#### 3.4.3 报告生成（`nanochat/report.py`）

**报告系统**：

```python
# 各个脚本调用
get_report().log(section="训练阶段", data={
    "参数": "值",
    "指标": 0.123,
})

# 最终生成
python -m nanochat.report generate
# 输出：report.md（包含所有阶段的汇总）
```

**报告内容**：
- 系统信息（GPU、内存、代码统计）
- 各阶段配置和结果
- 评估指标表格
- 训练时长和成本估算

---

## 四、训练流程详解

### 4.1 完整训练流程（speedrun.sh）

**步骤拆解**：

```bash
# ============ 环境准备 ============
# 1. 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建虚拟环境
uv venv
source .venv/bin/activate

# 3. 安装依赖
uv sync --extra gpu  # 或 --extra cpu

# ============ 分词器 ============
# 4. 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 5. 编译 rustbpe
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 6. 下载数据并训练分词器
python -m nanochat.dataset -n 8           # 下载 8 个分片（~2B 字符）
python -m nanochat.dataset -n 240 &       # 后台下载更多（预训练需要）
python -m scripts.tok_train --max_chars=2000000000  # 训练
python -m scripts.tok_eval                # 评估

# ============ 预训练 ============
# 7. 等待数据下载完成
wait $DATASET_DOWNLOAD_PID

# 8. 基础模型训练
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# 9. 评估基础模型
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss   # Bits per byte
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval   # CORE score

# ============ 中间训练 ============
# 10. 下载身份对话数据
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 11. Midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# ============ 监督微调 ============
# 12. SFT
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# ============ 推理 ============
# 13. 命令行聊天
python -m scripts.chat_cli -p "Why is the sky blue?"

# 14. Web 界面
python -m scripts.chat_web
# 访问 http://localhost:8000

# ============ 可选：强化学习 ============
# 15. RL（仅 GSM8K）
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K

# ============ 报告生成 ============
# 16. 生成完整报告
python -m nanochat.report generate
# 输出：report.md
```

### 4.2 训练时间和成本

**d20 模型（speedrun.sh，561M 参数）**：

```
设备：8XH100 GPU
训练时长：~4 小时
成本：~$100（$24/小时 × 4 小时）

阶段分解：
- 分词器训练：10 分钟
- 数据下载：20 分钟（后台）
- 预训练：2.5 小时
- Midtraining：30 分钟
- SFT：30 分钟
- 评估：30 分钟

训练 token 数：
- 预训练：~11B tokens（20×参数量，Chinchilla）
- Midtraining：~500M tokens
- SFT：~23K 样本 × 平均长度

性能：
- CORE score：~0.22
- MFU：~40%（H100 理论性能的 40%）
```

**d26 模型（更大模型）**：

```
设备：8XH100 GPU
训练时长：~12 小时
成本：~$300

参数量：~1.2B
训练 token 数：~24B tokens

性能：
- CORE score：~0.26（超越 GPT-2）
```

**d32 模型（run1000.sh，1.9B 参数）**：

```
设备：8XH100 GPU
训练时长：~33 小时
成本：~$800

训练 token 数：~38B tokens

性能：
- CORE score：更高（具体看 nanochat.karpathy.ai）
```

### 4.3 超参数调优建议

**模型规模缩放**：

```python
# 规则：
# - n_embd = depth × 64（可调到 128）
# - n_head = ceil(n_embd / 128)
# - head_dim = 128（固定）

# 示例：
depth = 20 -> n_embd = 1280, n_head = 10, params = 561M
depth = 26 -> n_embd = 1664, n_head = 13, params = 1.2B
depth = 32 -> n_embd = 2048, n_head = 16, params = 1.9B
```

**训练 token 数**：

```python
# Chinchilla 最优：tokens = 20 × params
# 可用范围：10-30 × params

# 计算所需分片数：
tokens_needed = params * 20
chars_needed = tokens_needed * 4.8  # 假设 4.8 chars/token
shards_needed = chars_needed / 250e6  # 每分片 250M 字符
```

**批次大小**：

```python
# 总批次大小（推荐）：
# - 小模型（<1B）：524,288 tokens
# - 大模型（1-3B）：1,048,576 tokens

# 设备批次大小（根据 VRAM 调整）：
# - 80GB GPU：32（depth=20）-> 16（depth=26）-> 8（depth=32）
# - 40GB GPU：减半
# - 单 GPU：尽可能大，代码会自动使用梯度累积
```

**学习率**：

```python
# 基本不需要调整，代码会根据 d_model 自动缩放
# 但如果必须调整：
# - 增大模型 -> 自动降低 LR（∝1/√d_model）
# - 增大批次 -> 可线性增大 LR（但代码已优化）
```

### 4.4 常见问题和解决方案

#### 问题 1：OOM（显存不足）

```bash
# 解决方案：
# 1. 减小 device_batch_size
torchrun ... --device_batch_size=16  # 从 32 减半

# 2. 减小序列长度
torchrun ... --max_seq_len=1024  # 从 2048 减半

# 3. 减小模型规模
torchrun ... --depth=16  # 从 20 减小

# 4. 启用梯度检查点（需要修改代码）
```

#### 问题 2：训练速度慢

```bash
# 检查：
# 1. MFU（Model FLOPs Utilization）
#    - 目标：>30%（H100）
#    - 如果低：可能是数据加载瓶颈

# 2. 数据加载优化
#    - 增加 tokenizer_threads（默认 4）
#    - 增加 tokenizer_batch_size（默认 128）

# 3. 编译优化
#    - 确保使用 torch.compile
#    - 可尝试 dynamic=True/False
```

#### 问题 3：分词器压缩率低

```bash
# 原因：
# - 训练数据太少
# - vocab_size 太小

# 解决：
# 1. 增加训练数据
python -m nanochat.dataset -n 16  # 从 8 增加到 16

# 2. 增加 vocab_size（需要重新训练）
python -m scripts.tok_train --vocab_size=100000
```

#### 问题 4：评估指标不提升

```bash
# 调试步骤：
# 1. 检查训练损失是否下降
#    - 如果不下降：学习率或优化器问题

# 2. 检查验证损失
#    - 如果下降但指标不升：可能是评估代码问题

# 3. 检查样本输出
#    - 使用 sample_every 查看生成质量

# 4. 增加训练时长
#    - 小模型需要更多数据才能学会任务
```

---

## 五、使用指南

### 5.1 快速开始

#### 5.1.1 环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# 2. 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 创建虚拟环境
uv venv

# 4. 激活环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 5. 安装依赖
uv sync --extra gpu        # GPU 版本
# 或
uv sync --extra cpu        # CPU 版本
```

#### 5.1.2 运行 speedrun（推荐）

```bash
# 在 8XH100 节点上：
bash speedrun.sh

# 或在 screen 会话中：
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# 监控进度（另一个终端）：
tail -f speedrun.log

# 分离 screen：Ctrl-A D
# 重新连接：screen -r speedrun
```

#### 5.1.3 仅推理（使用预训练模型）

```bash
# 1. 下载预训练模型（假设可用）
# wget https://... -O ~/.cache/nanochat/base_checkpoints/d20/

# 2. 下载分词器
# wget https://... -O ~/.cache/nanochat/tokenizer/

# 3. 命令行聊天
python -m scripts.chat_cli -p "你好，介绍一下自己"

# 4. Web 界面
python -m scripts.chat_web
# 访问 http://localhost:8000
```

### 5.2 自定义训练

#### 5.2.1 训练更小的模型（CPU/MPS）

```bash
# 参考 dev/runcpu.sh
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --num_iterations=20 \
    --eval_tokens=512 \
    --core_metric_every=-1
```

#### 5.2.2 自定义数据集

**添加自定义任务**：

```python
# tasks/my_task.py

from tasks.common import Task

class MyTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 加载你的数据
        self.data = load_my_data()
    
    @property
    def eval_type(self):
        return "categorical"  # 或 "generative"
    
    def num_examples(self):
        return len(self.data)
    
    def get_example(self, index):
        # 返回对话格式：
        # {
        #     "messages": [
        #         {"role": "user", "content": "问题"},
        #         {"role": "assistant", "content": "答案"},
        #     ]
        # }
        item = self.data[index]
        return {
            "messages": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]
        }
    
    def evaluate(self, problem, completion):
        # 生成任务的评估逻辑
        return completion.strip() == problem["answer"].strip()
```

**在 SFT 中使用**：

```python
# scripts/chat_sft.py（修改）

from tasks.my_task import MyTask

train_ds = TaskMixture([
    ARC(...),
    MyTask(split="train"),  # 添加你的任务
    # ...
])
```

#### 5.2.3 自定义身份/个性

```python
# 1. 创建身份对话数据
# dev/gen_synthetic_data.py（参考）

conversations = [
    {
        "messages": [
            {"role": "user", "content": "你是谁？"},
            {"role": "assistant", "content": "我是 MyBot，一个专注于数学辅导的 AI 助手。"},
        ]
    },
    # ... 更多对话
]

# 保存为 JSONL
import json
with open("my_identity.jsonl", "w") as f:
    for conv in conversations:
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

# 2. 在训练中使用
# scripts/chat_sft.py（修改）
identity_conversations_filepath = "my_identity.jsonl"
train_ds = TaskMixture([
    # ...
    CustomJSON(filepath=identity_conversations_filepath),
])
```

### 5.3 评估和调试

#### 5.3.1 单独评估模型

```bash
# 基础模型 CORE 评估
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# 聊天模型评估（所有任务）
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# 聊天模型评估（特定任务）
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a GSM8K

# 单 GPU 评估
python -m scripts.chat_eval -i sft -a MMLU
```

#### 5.3.2 调试分词器

```python
# 1. 检查分词效果
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()

text = "Hello, world! 你好世界"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")

# 2. 可视化对话渲染
conversation = {
    "messages": [
        {"role": "user", "content": "测试"},
        {"role": "assistant", "content": "好的"},
    ]
}

ids, mask = tokenizer.render_conversation(conversation)
print(tokenizer.visualize_tokenization(ids, mask))
# 红色=不训练，绿色=训练
```

#### 5.3.3 调试模型生成

```python
# scripts/debug_generate.py（自己创建）

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init
from nanochat.engine import Engine

# 初始化
_, _, _, _, device = compute_init()
model, tokenizer, _ = load_model("sft", device)

# 创建引擎
engine = Engine(model, tokenizer)

# 生成
prompt = "为什么天空是蓝色的？"
tokens = tokenizer.encode(prompt, prepend="<|bos|>")
tokens.append(tokenizer.encode_special("<|user_start|>"))
tokens.extend(tokenizer.encode(prompt))
tokens.append(tokenizer.encode_special("<|user_end|>"))
tokens.append(tokenizer.encode_special("<|assistant_start|>"))

print("Prompt:", tokenizer.decode(tokens))
print("\nGenerating...")

for token_column, token_masks in engine.generate(tokens, num_samples=1, max_tokens=100, temperature=0.7):
    token = token_column[0]
    chunk = tokenizer.decode([token])
    print(chunk, end="", flush=True)

print()
```

### 5.4 部署

#### 5.4.1 Web 服务部署

```bash
# 1. 本地开发
python -m scripts.chat_web
# 默认：http://localhost:8000

# 2. 指定端口
python -m scripts.chat_web --port 8080

# 3. 允许外部访问
python -m scripts.chat_web --host 0.0.0.0 --port 8000

# 4. 生产环境（使用 gunicorn）
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker scripts.chat_web:app
```

#### 5.4.2 CLI 工具

```bash
# 交互式聊天
python -m scripts.chat_cli

# 单次问答
python -m scripts.chat_cli -p "问题"

# 指定模型
python -m scripts.chat_cli -i mid  # 使用 midtrain 模型
python -m scripts.chat_cli -i sft  # 使用 SFT 模型（默认）
```

#### 5.4.3 API 集成

```python
# your_app.py

from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.common import compute_init

class NanoChatAPI:
    def __init__(self):
        _, _, _, _, device = compute_init()
        model, tokenizer, _ = load_model("sft", device)
        self.engine = Engine(model, tokenizer)
        self.tokenizer = tokenizer
    
    def chat(self, message, history=None):
        """
        message: str - 用户消息
        history: List[Dict] - 历史对话（可选）
        
        返回：str - 助手回复
        """
        # 构造对话
        messages = history or []
        messages.append({"role": "user", "content": message})
        
        # 渲染为 token
        conversation = {"messages": messages}
        tokens = self.tokenizer.render_for_completion(conversation)
        
        # 生成
        completion_tokens, _ = self.engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=512,
            temperature=0.7,
        )
        
        # 解码
        reply = self.tokenizer.decode(completion_tokens[0])
        
        # 去除特殊 token
        reply = reply.replace("<|assistant_end|>", "").strip()
        
        return reply

# 使用
api = NanoChatAPI()
response = api.chat("你好")
print(response)
```

---

## 六、总结与最佳实践

### 6.1 项目核心价值

**nanochat 的独特之处**：

1. **教学友好**：
   - 代码量适中（~8K 行）
   - 注释详尽，逻辑清晰
   - 避免过度抽象和配置复杂性

2. **端到端完整**：
   - 从数据下载到 Web 服务的完整流程
   - 每个阶段都有独立脚本，可单独运行
   - 自动化程度高（speedrun.sh 一键训练）

3. **成本可控**：
   - $100 快速验证
   - $300-$1000 达到实用水平
   - 对学习者和小团队友好

4. **现代技术**：
   - RoPE、QK Norm、MQA/GQA
   - Muon 优化器（二阶方法）
   - KV Cache、Flash Attention
   - 工具调用能力

### 6.2 最佳实践

#### 6.2.1 开发流程

```
1. 小规模验证
   ├─→ 在 CPU/单 GPU 上训练小模型（depth=4）
   ├─→ 验证代码逻辑正确
   └─→ 快速迭代（几分钟到几小时）

2. 中等规模实验
   ├─→ 使用单 GPU 训练 d12-d16 模型
   ├─→ 调整超参数
   └─→ 评估效果（几小时到一天）

3. 全规模训练
   ├─→ 使用 8XH100 训练 d20-d32 模型
   ├─→ 运行完整流程（speedrun.sh）
   └─→ 生成报告和部署（几小时到几天）
```

#### 6.2.2 代码修改建议

**修改前必读**：

1. **引用检查**：
   ```bash
   # 使用 IDE 的"查找引用"功能
   # 或使用 grep
   grep -r "function_name" nanochat/ scripts/
   ```

2. **保持完整性**：
   - 修改函数时，提供完整代码
   - 不要只给片段（除非代码审查）

3. **同步修改**：
   - 如果修改了函数签名，更新所有调用点
   - 如果修改了配置，更新默认值和文档

4. **测试**：
   ```bash
   # 运行测试
   pytest tests/test_rustbpe.py -v
   
   # 小规模验证
   python -m scripts.base_train --depth=4 --num_iterations=10
   ```

#### 6.2.3 性能优化建议

**训练速度**：

1. **数据加载**：
   - 增加 `tokenizer_threads`（4 -> 8）
   - 增加 `tokenizer_batch_size`（128 -> 256）
   - 确保数据已预下载

2. **计算效率**：
   - 使用 `torch.compile`（已启用）
   - 使用 bfloat16（已启用）
   - 启用 TF32（已启用）
   - 目标 MFU > 30%

3. **内存优化**：
   - 降低 `device_batch_size`，增加 `grad_accum_steps`
   - 使用梯度检查点（需要添加）
   - 清理不需要的中间结果

**推理速度**：

1. **KV Cache**：
   - 已启用（`Engine` 类）
   - 确保使用 `Engine.generate` 而非 `model.generate`

2. **批量推理**：
   ```python
   # 利用 num_samples 参数
   completions = engine.generate_batch(
       tokens,
       num_samples=8,  # 并行生成 8 个样本
       ...
   )
   ```

3. **量化（未实现，可添加）**：
   - int8 量化
   - int4 量化
   - 需要修改模型代码

### 6.3 扩展方向

**可能的改进**：

1. **模型架构**：
   - 添加 MoE（Mixture of Experts）
   - 尝试其他激活函数（SwiGLU）
   - 实验不同的归一化方法

2. **训练方法**：
   - 添加更多 RL 算法（PPO、DPO）
   - 实现课程学习
   - 多阶段学习率调整

3. **数据质量**：
   - 数据去重
   - 数据过滤（毒性、质量）
   - 更多合成数据

4. **推理优化**：
   - 模型量化
   - 投机解码（Speculative Decoding）
   - 批量动态调度

5. **工具能力**：
   - 添加更多工具（搜索、文件操作）
   - 多轮工具调用
   - 视觉输入（多模态）

### 6.4 学习路径

**对于初学者**：

```
第 1 周：理解基础
├─→ 阅读 README.md
├─→ 运行 speedrun.sh（如果有 GPU）
│   或 dev/runcpu.sh（如果没有 GPU）
└─→ 查看生成的 report.md

第 2 周：理解代码
├─→ 阅读 nanochat/gpt.py（模型定义）
├─→ 阅读 nanochat/engine.py（推理引擎）
└─→ 阅读 scripts/base_train.py（训练循环）

第 3 周：修改实验
├─→ 修改模型超参数（depth, n_head）
├─→ 添加自定义数据集
└─→ 调整训练配置

第 4 周：深入优化
├─→ 研究优化器（Muon vs AdamW）
├─→ 分析性能（MFU, 内存使用）
└─→ 实验新想法
```

**对于进阶用户**：

```
研究方向 1：架构创新
├─→ 实现新的注意力机制
├─→ 实验模型压缩技术
└─→ 对比不同设计选择

研究方向 2：训练优化
├─→ 实现新的优化算法
├─→ 研究学习率调度
└─→ 数据混合策略

研究方向 3：应用扩展
├─→ 多模态扩展（视觉）
├─→ 长文本支持（>2048 tokens）
└─→ 特定领域适应
```

### 6.5 常见陷阱

**避免这些错误**：

1. **过早优化**：
   - ❌ 先调整优化器参数
   - ✅ 先确保基本流程跑通

2. **忽视数据质量**：
   - ❌ 只关注模型大小
   - ✅ 数据质量 > 模型大小

3. **评估不充分**：
   - ❌ 只看训练损失
   - ✅ 多样化评估（CORE, 人工检查）

4. **依赖过多**：
   - ❌ 添加大量外部库
   - ✅ 保持代码简洁

5. **缺乏文档**：
   - ❌ 修改代码不留注释
   - ✅ 详细记录修改原因

### 6.6 结语

**nanochat 的设计哲学**：

> "Simplicity is the ultimate sophistication."  
> —— Leonardo da Vinci

nanochat 的目标不是成为最强大或最灵活的 LLM 框架，而是成为**最易理解和修改的完整 LLM 实现**。通过牺牲一些灵活性和抽象性，我们获得了：

- **可读性**：任何有 PyTorch 经验的人都能读懂
- **可修改性**：想改什么就改什么，不需要理解复杂的抽象层
- **可复现性**：单个脚本，端到端，结果可复现
- **教学价值**：作为学习材料，比复杂框架更有价值

**最后建议**：

1. **动手实践**：不要只读代码，运行它，修改它，破坏它，修复它
2. **提问和分享**：在 Discussions 中提问，分享你的改进
3. **保持简洁**：添加新功能时，问自己"这真的必要吗？"
4. **享受过程**：训练 LLM 很有趣，享受这个过程！

---

## 附录

### A. 目录结构速查

```
nanochat/
├── nanochat/              # 核心库
│   ├── gpt.py            # ⭐ GPT 模型定义
│   ├── engine.py         # ⭐ 推理引擎
│   ├── dataloader.py     # ⭐ 数据加载
│   ├── tokenizer.py      # ⭐ 分词器
│   ├── muon.py           # Muon 优化器
│   ├── adamw.py          # AdamW 优化器
│   ├── checkpoint_manager.py  # 检查点管理
│   ├── common.py         # 工具函数
│   ├── configurator.py   # 配置系统
│   ├── dataset.py        # 数据下载
│   ├── core_eval.py      # CORE 评估
│   ├── loss_eval.py      # 损失评估
│   ├── execution.py      # 工具执行
│   ├── report.py         # 报告生成
│   └── ui.html           # Web UI
├── scripts/              # 可执行脚本
│   ├── tok_train.py      # 📝 分词器训练
│   ├── tok_eval.py       # 📝 分词器评估
│   ├── base_train.py     # 🚀 基础模型训练
│   ├── base_eval.py      # 🚀 基础模型评估
│   ├── base_loss.py      # 🚀 基础模型损失
│   ├── mid_train.py      # 💬 中间训练
│   ├── chat_sft.py       # 💬 监督微调
│   ├── chat_rl.py        # 🎯 强化学习
│   ├── chat_eval.py      # 💬 聊天评估
│   ├── chat_cli.py       # 💻 命令行界面
│   └── chat_web.py       # 🌐 Web 界面
├── tasks/                # 评估任务
│   ├── common.py         # 任务基类
│   ├── arc.py            # ARC 任务
│   ├── gsm8k.py          # 数学任务
│   ├── humaneval.py      # 代码任务
│   ├── mmlu.py           # MMLU 任务
│   ├── smoltalk.py       # 对话任务
│   ├── spellingbee.py    # 拼写任务
│   └── customjson.py     # 自定义任务
├── rustbpe/              # Rust 分词器
├── tests/                # 测试
├── dev/                  # 开发工具
├── speedrun.sh           # ⚡ 快速训练脚本
├── run1000.sh            # 💰 完整训练脚本
└── pyproject.toml        # 项目配置
```

### B. 命令速查表

```bash
# ============ 环境设置 ============
uv venv && source .venv/bin/activate
uv sync --extra gpu

# ============ 训练 ============
# 分词器
python -m scripts.tok_train --max_chars=2000000000

# 基础模型
torchrun --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Midtraining
torchrun --nproc_per_node=8 -m scripts.mid_train

# SFT
torchrun --nproc_per_node=8 -m scripts.chat_sft

# RL（可选）
torchrun --nproc_per_node=8 -m scripts.chat_rl

# ============ 评估 ============
# 基础模型
torchrun --nproc_per_node=8 -m scripts.base_eval

# 聊天模型
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# 特定任务
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a GSM8K

# ============ 推理 ============
# CLI
python -m scripts.chat_cli
python -m scripts.chat_cli -p "问题"

# Web
python -m scripts.chat_web

# ============ 工具 ============
# 下载数据
python -m nanochat.dataset -n 8

# 生成报告
python -m nanochat.report generate

# 测试
pytest tests/test_rustbpe.py -v
```

### C. 配置参数速查

**base_train.py 主要参数**：

```python
--depth=20              # 模型深度（层数）
--max_seq_len=2048      # 序列长度
--device_batch_size=32  # 每设备批次大小
--total_batch_size=524288  # 总批次大小
--num_iterations=5000   # 训练步数
--matrix_lr=0.02        # Muon 学习率
--embedding_lr=0.2      # 嵌入学习率
--eval_every=250        # 评估频率
--run=dummy             # wandb 运行名
```

**chat_sft.py 主要参数**：

```python
--source=mid            # 源模型（base/mid）
--device_batch_size=4   # 批次大小
--num_epochs=1          # 训练轮数
--matrix_lr=0.02        # 学习率
--init_lr_frac=0.02     # 初始学习率比例
--eval_every=100        # 评估频率
```

### D. 性能基准

**d20 模型（561M 参数）**：

```
训练：
- 时间：2.5 小时（8XH100）
- MFU：~40%
- Tokens/sec：~200K

评估：
- CORE：~0.22
- ARC-Easy：~0.36
- ARC-Challenge：~0.28
- GSM8K（SFT）：~0.05
- MMLU：~0.31

推理：
- Tokens/sec：~100（单GPU，batch=1）
- 延迟：~10ms/token
```

**d26 模型（1.2B 参数）**：

```
训练：
- 时间：~10 小时（8XH100）
- CORE：~0.26（超越 GPT-2）
```

### E. 资源链接

**官方资源**：
- GitHub：https://github.com/karpathy/nanochat
- Discussions：https://github.com/karpathy/nanochat/discussions
- Demo：https://nanochat.karpathy.ai

**相关项目**：
- nanoGPT：https://github.com/karpathy/nanoGPT
- modded-nanoGPT：https://github.com/KellerJordan/modded-nanogpt

**学习资源**：
- LLM101n 课程：（待发布）
- PyTorch 文档：https://pytorch.org/docs
- Andrej Karpathy YouTube：https://youtube.com/@AndrejKarpathy

---

**文档结束**

*如有疑问或建议，欢迎在 GitHub Discussions 中提出！*

*祝你训练愉快！🚀*

