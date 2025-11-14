#!/bin/bash

# ============================================================================
# Run1000脚本 - nanochat的$1000级别训练
# ============================================================================
#
# 设计目标：
#   在8XH100节点上以$24/小时的价格，端到端运行约41.6小时
#   总成本：$1000 / $24/hour ≈ 41.6 hours
#
# 与speedrun.sh的区别：
#   - 更大的模型：depth=32（1.88B参数）vs depth=20（561M参数）
#   - 更多训练数据：800个分片 vs 240个分片
#   - 更长训练时间：约31.3小时预训练 vs 约4小时总计
#   - 更高性能：预期更好的CORE和ChatCORE分数
#
# 注释较简洁，详细说明请参见speedrun.sh
#
# 模型规格（d32）：
#   - 参数数量：1,879,048,192（约1.88B）
#   - 层数：32
#   - 模型维度：2048
#   - 注意力头数：16
#   - 训练tokens：约37.6B（20X参数数，Chinchilla定律）
#   - 训练字符：约185B（假设4.8 chars/token）
#   - 需要数据分片：约740个（250M chars/shard）
#   - 实际下载：800个（安全余量）
# ============================================================================

# 环境设置（与speedrun.sh相同，详细注释见speedrun.sh）
export OMP_NUM_THREADS=1
# 限制OpenMP线程数为1，避免与PyTorch线程竞争导致性能下降

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# 设置nanochat基础目录，所有中间产物保存在这里

mkdir -p $NANOCHAT_BASE_DIR
# -p：如果目录已存在不报错，自动创建所有必需的父目录

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# 检查uv是否安装，如果没有则下载安装脚本并执行
# command -v uv：返回uv路径或空（未安装）
# &> /dev/null：丢弃输出
# ||：如果uv不存在，执行右侧命令（下载安装）

[ -d ".venv" ] || uv venv
# 检查.venv目录是否存在，不存在则创建虚拟环境
# [ -d ".venv" ]：测试目录是否存在
# ||：目录不存在时执行uv venv

uv sync --extra gpu
# 同步项目依赖，安装pyproject.toml中定义的所有包
# --extra gpu：额外安装GPU组的依赖（CUDA版PyTorch等）

source .venv/bin/activate
# 激活虚拟环境，修改PATH使python指向.venv/bin/python

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
# 如果WANDB_RUN未设置（空字符串），则设为"dummy"
# "dummy"是特殊值：训练脚本会跳过wandb日志记录

python -m nanochat.report reset
# 重置训练报告：清空report/目录并生成新的报告头部
# 包含系统信息、Git状态、GPU信息、开始时间戳等

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# 下载并安装Rust工具链（非交互式）
# --proto '=https'：仅https协议（安全）
# --tlsv1.2：TLS 1.2+
# -y：自动确认所有提示

source "$HOME/.cargo/env"
# 加载Rust环境变量，将cargo（Rust包管理器）加入PATH

uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
# 编译并安装rustbpe分词器到当前虚拟环境
# maturin：Rust-Python绑定构建工具
# develop：开发模式（构建并安装）
# --release：发布模式（优化编译，速度最快）
# --manifest-path：指定Rust项目配置文件Cargo.toml的路径

curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# 下载合成身份对话文件（2.3MB，约100个对话）
# -L：跟随重定向
# -o：指定输出文件路径
# 这些对话定义了nanochat的"个性"和回答风格

# 在约4B字符上训练分词器，并启动剩余数据下载用于预训练
python -m nanochat.dataset -n 16
# ├─ python -m nanochat.dataset：运行数据集下载模块
# └─ -n 16：下载前16个数据分片
#     16个分片 × 250M chars/分片 = 4B chars（用于训练分词器）

# 开始下载剩余分片，总共800个（下文解释为什么是800）
python -m nanochat.dataset -n 800 &
# ├─ -n 800：下载前800个分片（约200GB压缩数据）
# ├─ &：后台运行，不阻塞当前shell
# └─ 说明：d32模型需要约185B chars（740个分片），为安全起见下载800个

# TODO: 下载其余部分（注释说明：如果需要更多数据可继续下载）

python -m scripts.tok_train --max_chars=4000000000
# 训练BPE分词器
# --max_chars=4000000000：在4B字符上训练（前16个分片）
# 词汇表大小：65536（2^16），在代码中固定

python -m scripts.tok_eval
# 评估分词器性能
# 输出：bits per byte（压缩比）、chars/token（约4.8）、分词示例

# ============================================================================
# 超参数确定过程详细记录
# ============================================================================
# 目标：约$1000预算 ≈ 41.6小时的8XH100计算
#
# 1) 模型规模选择：
#    我猜测这个预算下的模型大小约为depth=32
#
# 2) 确定可容纳的device_batch_size：
#    使用--depth=32运行base_train.py脚本，我发现--device_batch_size=16会耗尽内存，
#    但--device_batch_size=8可以容纳。训练期间检查`nvidia-smi`，
#    我看到所有GPU都在约78/80GB VRAM，所以刚好容纳，并且MFU约50%很好。
#    
#    训练脚本运行正常并显示：
#    - 词汇表大小: 65,536
#    - 层数 (num_layers): 32
#    - 模型维度 (model_dim): 2048
#    - 注意力头数 (num_heads): 16
#    - KV头数 (num_kv_heads): 16
#    - Tokens / micro-batch / rank: 8 x 2048 = 16,384
#    - Tokens / micro-batch: 131,072
#    - 总批量大小 524,288 => 梯度累积步数: 4
#    - 参数数量: 1,879,048,192
#    - 每token估计FLOPs: 1.207960e+10
#    - 根据目标data:param比率计算的迭代次数: 71,680
#    - 训练tokens总数: 37,580,963,840
#    - Tokens : Params 比率: 20.00
#    - 训练FLOPs总估计: 4.539628e+20
#    - step 00004/71680 (0.01%) | loss: 8.813754 | lrm: 1.00 | dt: 1571.88ms | tok/sec: 83,385 | mfu: 50.92 | total time: 0.00m
#    - step 00005/71680 (0.01%) | loss: 8.488074 | lrm: 1.00 | dt: 1572.76ms | tok/sec: 83,338 | mfu: 50.89 | total time: 0.00m
#
# 3) 验证运行时间是否符合预算：
#    训练脚本使用Chinchilla缩放定律以计算最优方式设置#tokens = 20 * #params。具体来说：
#    脚本显示我们将训练71,680步，每步需要1.574秒，所以：
#    估计训练时间: 71,680 * 1.574s / 60 / 60 = 31.3小时
#    这是可以的，符合我们的预算，并为midtraining、SFT、evals和可能的RL留下约10小时。
#    可能depth=33或depth=34也能容纳，但现在让我们继续使用这个。
#
# 4) 需要关注的最后一件事是运行所需的训练数据量：
#    上面的脚本计算出"训练tokens总数: 37,580,963,840"
#    tok_eval.py脚本报告默认分词器设置下平均约4.8 chars/token。
#    所以约38B tokens × 约4.8 chars/token = 约185B chars。
#    每个数据分片约250M chars，所以我们需要约185B / 250M ≈ 740个分片。
#    为安全起见，我将其提高到800个分片，这就是为什么上面预下载数据集分片时使用-n 800。
#    
#    如果我们没有足够的数据，训练脚本会循环并在相同数据上进行多个epoch，
#    这会降低模型性能。可能2、3个epoch左右还可以，但肯定不理想，
#    在10+个epoch时我们会开始严重过拟合。
#
# 5) 就是这样，其他所有内容（例如学习率）都由训练脚本自动调整。
# ============================================================================

# 使用的进程数/GPU数
NPROC_PER_NODE=8
# 8个进程 = 8个GPU（每个GPU一个进程）

# 基础预训练（depth=32，device_batch_size=8）
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=32 --device_batch_size=8 --run=$WANDB_RUN
# ├─ torchrun：PyTorch分布式训练启动器
# ├─ --standalone：单节点模式（所有进程在同一台机器）
# ├─ --nproc_per_node=8：每个节点8个进程（对应8个GPU）
# ├─ -m scripts.base_train：运行base_train模块
# ├─ --：分隔torchrun参数和脚本参数
# └─ 脚本参数：
#     ├─ --depth=32：模型深度
#     │   d32模型：44层（32+12），1.88B参数
#     │   训练时间：约31.3小时（71,680步 × 1.574秒/步）
#     ├─ --device_batch_size=8：每个GPU的批量大小
#     │   总批量 = 8 GPU × 8 batch × 2048 seq_len × 4 grad_accum = 524,288 tokens
#     │   说明：device_batch_size=16会OOM，8刚好合适（78/80GB VRAM）
#     └─ --run=$WANDB_RUN：WandB运行名称

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# 评估训练后的base模型
# 在更大数据块上计算训练集和验证集的BPB（bits per byte）
# 生成文本样本展示模型能力

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
# 在CORE基准测试上评估base模型
# 评估任务：ARC、HellaSwag、MMLU、TruthfulQA、Winogrande等
# 计算CORE分数（综合推理能力指标）

# 中期训练（Midtrain）
# 注意：确保这里使用与基础训练脚本相同的device_batch_size
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --device_batch_size=8 --run=$WANDB_RUN
# ├─ scripts.mid_train：中期训练脚本
# ├─ --device_batch_size=8：必须与base_train保持一致（避免OOM）
# └─ 作用：
#     - 继续训练base模型，教它学会：
#       * 对话格式（特殊token：<|user_start|>等）
#       * 工具使用（Python计算器工具调用）
#       * 多选题格式
#       * SpellingBee/SimpleSpelling任务
#       * Identity对话（个性化）
#     - 学习率较低，约训练1-2个epoch
#     - 预期时间：约2-3小时

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
# ├─ scripts.chat_eval：聊天模型评估脚本
# ├─ -i mid：input model tag，评估mid模型
# └─ 作用：
#     - 评估对话任务：GSM8K、HumanEval、MMLU、ARC
#     - 计算ChatCORE分数
#     - 生成详细评估报告

# 监督微调（SFT）
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
# ├─ scripts.chat_sft：监督微调脚本
# └─ 作用：
#     - 在混合任务数据集上微调：
#       * ARC-Easy/Challenge（推理）
#       * GSM8K（数学）
#       * SmolTalk（通用对话）
#       * CustomJSON（身份对话）
#       * SpellingBee（拼写）
#     - 训练1-2个epoch
#     - 每个任务独立优化，提升特定领域能力
#     - 预期时间：约2-3小时

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
# ├─ -i sft：评估sft模型（监督微调后）
# └─ 预期：相比mid模型，各任务性能应有1-3%的提升

# 生成最终报告
python -m nanochat.report generate
# ├─ 收集所有训练阶段的报告片段
# ├─ 组合成完整的Markdown报告
# └─ 输出位置：
#     * $NANOCHAT_BASE_DIR/report/report.md（原始）
#     * ./report.md（当前目录，方便查看）
# 报告内容：
#     - 系统信息（硬件、软件、Git状态）
#     - 分词器性能（BPB、chars/token）
#     - 训练指标（loss曲线、MFU、训练时间）
#     - 评估结果（CORE、ChatCORE分数、各任务准确率）
#     - 文本样本（展示模型能力）
#     - 成本估算（基于GPU小时数和价格）

# 与模型对话（启动Web界面）
python -m scripts.chat_web
# ├─ scripts.chat_web：Web聊天服务器
# └─ 功能：
#     - 启动FastAPI服务器（默认端口8000）
#     - 提供ChatGPT风格的Web界面
#     - 支持流式输出（实时显示生成文本）
#     - 多GPU并行推理（worker pool）
#     - 滥用防护（限制消息长度、对话长度、生成参数）
#     - 访问：http://localhost:8000
# 使用方式：
#     - 打开浏览器访问http://localhost:8000
#     - 输入消息，按Enter或点击发送
#     - 支持多轮对话（保持上下文）
#     - Ctrl+C停止服务器
