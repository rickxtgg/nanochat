#!/bin/bash

# ============================================================================
# Speedrun脚本 - "100美元能买到的最好的ChatGPT克隆"
# ============================================================================
#
# 设计目标：
#   在8XH100节点上以$3/GPU/小时的价格，约4小时内完成全部训练
#   总成本：8 GPUs × $3/hour × 4 hours = $96 ≈ $100
#
# 运行方式：
#   1) 最简单的启动方式：
#      bash speedrun.sh
#
#   2) 在screen会话中启动（推荐，因为运行需要约4小时）：
#      screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
#
#   3) 启用wandb日志记录（需先设置wandb，见下文）：
#      WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
#
# 训练阶段：
#   1. 分词器训练（Tokenizer）
#   2. 基础预训练（Base Pre-training）- d20模型，561M参数
#   3. 中期训练（Midtraining）- 对话格式和工具使用
#   4. 监督微调（SFT）- 任务适配
#   5. 强化学习（RL）- 可选，仅GSM8K
#
# 输出：
#   - 中间产物保存在~/.cache/nanochat
#   - 最终报告：report.md
#   - 可通过chat_cli或chat_web与模型交互
# ============================================================================

# 环境变量配置
# export命令：设置环境变量，使其在当前shell及其所有子进程中可用
export OMP_NUM_THREADS=1  
# ├─ OMP_NUM_THREADS=1：限制OpenMP（并行计算库）使用的线程数为1
# └─ 原因：避免PyTorch和OpenMP的线程竞争，防止CPU过载导致性能下降

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# ├─ NANOCHAT_BASE_DIR：nanochat项目的基础目录路径
# ├─ $HOME：当前用户的主目录（如/home/username）
# └─ .cache/nanochat：在主目录下的隐藏缓存目录

# mkdir命令：创建目录
mkdir -p $NANOCHAT_BASE_DIR
# ├─ mkdir：make directory（创建目录）
# ├─ -p：parents（父目录），如果目录已存在不报错，且自动创建所有必需的父目录
# └─ $NANOCHAT_BASE_DIR：使用上面定义的环境变量作为目录路径

# -----------------------------------------------------------------------------
# Python虚拟环境设置（使用uv）
# -----------------------------------------------------------------------------

# 安装uv（如果尚未安装）
# uv是一个快速的Python包管理器，比pip快10-100倍
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# ├─ command -v uv：检查uv命令是否存在（返回uv的路径或空）
# ├─ &> /dev/null：将标准输出和标准错误重定向到/dev/null（丢弃输出）
# ├─ ||：逻辑OR，如果左侧命令失败（uv不存在），则执行右侧命令
# ├─ curl：下载工具
# │   ├─ -L：follow redirects（跟随重定向）
# │   ├─ -s：silent（静默模式，不显示进度）
# │   ├─ -S：show error（显示错误，与-s配合使用）
# │   └─ -f：fail silently（HTTP错误时静默失败）
# └─ | sh：将下载的安装脚本通过管道传给shell执行

# 创建.venv本地虚拟环境（如果不存在）
[ -d ".venv" ] || uv venv
# ├─ [ -d ".venv" ]：测试.venv目录是否存在
# │   ├─ [：test命令的别名
# │   ├─ -d：directory（检查是否为目录）
# │   └─ ".venv"：虚拟环境目录名
# ├─ ||：如果左侧为假（目录不存在），执行右侧命令
# └─ uv venv：创建新的Python虚拟环境

# 安装项目依赖（GPU版本）
uv sync --extra gpu
# ├─ uv sync：同步项目依赖，根据pyproject.toml和uv.lock安装包
# └─ --extra gpu：安装额外的GPU相关依赖（如torch的CUDA版本）
#     说明：pyproject.toml中定义了[project.optional-dependencies]
#           gpu组包含了CUDA版本的PyTorch等GPU加速库

# 激活虚拟环境，使`python`使用项目的venv而非系统Python
source .venv/bin/activate
# ├─ source：在当前shell中执行脚本（而非启动新shell）
# └─ .venv/bin/activate：激活脚本，修改PATH等环境变量
#     作用：将.venv/bin加入PATH开头，使python命令指向虚拟环境

# -----------------------------------------------------------------------------
# Weights & Biases (WandB) 日志设置
# -----------------------------------------------------------------------------
# 如果希望使用wandb进行日志记录（很好用！推荐）：
# 1) 首先确保登录wandb，例如运行：
#    `wandb login`
# 2) 运行此脚本时设置WANDB_RUN环境变量，例如：
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
# ├─ if：条件判断语句
# ├─ [ -z "$WANDB_RUN" ]：测试WANDB_RUN是否为空字符串
# │   ├─ -z：zero length（零长度，即空字符串）
# │   └─ "$WANDB_RUN"：引号确保变量未定义时不会导致语法错误
# └─ then：如果条件为真，执行以下命令
    # 默认使用"dummy"：作为特殊情况处理，跳过wandb日志记录
    WANDB_RUN=dummy
    # ├─ 设置WANDB_RUN=dummy（不使用export，仅当前脚本可见）
    # └─ "dummy"是特殊值：训练脚本检测到这个值会跳过wandb日志
fi
# └─ fi：结束if语句（if倒过来写）

# -----------------------------------------------------------------------------
# 训练报告初始化
# -----------------------------------------------------------------------------
# 在运行过程中，我们会将markdown报告写入基础目录中的report/目录。
# 此命令清空它并写入头部部分，包含大量系统信息和标记运行开始的时间戳。
python -m nanochat.report reset
# ├─ python：Python解释器（来自激活的虚拟环境）
# ├─ -m：module（以模块方式运行），在sys.path中查找并执行模块
# ├─ nanochat.report：模块路径（对应nanochat/report.py）
# └─ reset：传给report模块的命令行参数
#     作用：清空$NANOCHAT_BASE_DIR/report/目录并生成新的报告头部
#           包含：系统信息、Git信息、GPU信息、时间戳等

# -----------------------------------------------------------------------------
# 分词器（Tokenizer）训练
# -----------------------------------------------------------------------------

# 安装Rust / Cargo（Rust工具链）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# ├─ curl：下载工具
# │   ├─ --proto '=https'：仅允许https协议（安全）
# │   ├─ --tlsv1.2：使用TLS 1.2或更高版本
# │   ├─ -s：silent（静默模式）
# │   ├─ -S：show error（显示错误）
# │   └─ -f：fail silently（HTTP错误时失败）
# ├─ | sh -s --：将下载的脚本传给shell执行
# │   ├─ sh：shell解释器
# │   ├─ -s：从标准输入读取命令
# │   └─ --：后面的参数传给脚本
# └─ -y：yes（自动确认所有提示，非交互式安装）

source "$HOME/.cargo/env"
# ├─ source：在当前shell中执行脚本
# └─ $HOME/.cargo/env：Rust环境变量配置脚本
#     作用：将Cargo（Rust包管理器）的bin目录加入PATH

# 构建rustbpe分词器（使用Rust实现，高性能）
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
# ├─ uv run：在虚拟环境中运行命令
# ├─ maturin：Rust-Python绑定构建工具
# ├─ develop：开发模式构建（构建并安装到当前环境）
# ├─ --release：发布模式（优化编译，速度快但编译慢）
# │   对比：--debug模式编译快但运行慢
# └─ --manifest-path rustbpe/Cargo.toml：指定Rust项目的清单文件路径
#     Cargo.toml是Rust项目的配置文件，类似Python的pyproject.toml

# 下载预训练数据集的前约2B字符
# 详见dev/repackage_data_reference.py了解数据准备细节
# 每个数据分片约250M字符
# 所以我们下载 2e9 / 250e6 = 8 个数据分片
# 每个分片约100MB文本（压缩后），总共约800MB磁盘数据
python -m nanochat.dataset -n 8
# ├─ python -m nanochat.dataset：运行数据集下载模块
# └─ -n 8：参数，下载前8个数据分片
#     说明：数据集是FineWeb-Edu-100BT，已预先分片为1822个Parquet文件
#           每个分片约250M字符，100MB压缩大小

# 立即在后台启动下载更多分片，同时训练分词器
# 下文解释为什么这里是240
python -m nanochat.dataset -n 240 &
# ├─ python -m nanochat.dataset -n 240：下载前240个分片
# ├─ &：后台运行（不阻塞当前shell，与下面的命令并行执行）
# └─ 说明：这样可以在训练分词器的同时下载数据，节省时间

DATASET_DOWNLOAD_PID=$!
# ├─ $!：特殊变量，保存最后一个后台进程的PID（进程ID）
# └─ 作用：保存下载进程的PID，以便后面用wait命令等待其完成

# 在约2B字符的数据上训练分词器，词汇表大小为2**16 = 65536
python -m scripts.tok_train --max_chars=2000000000
# ├─ python -m scripts.tok_train：运行分词器训练脚本
# └─ --max_chars=2000000000：最大训练字符数（2B = 2,000,000,000）
#     说明：在前8个分片（约2B字符）上训练BPE分词器
#           词汇表大小固定为65536（2^16），在代码中硬编码

# 评估分词器（报告压缩比等）
python -m scripts.tok_eval
# ├─ python -m scripts.tok_eval：运行分词器评估脚本
# └─ 作用：
#     - 计算压缩比（bits per byte）
#     - 计算平均chars/token（通常约4.8）
#     - 生成样本文本的分词可视化
#     - 结果记录到report

# -----------------------------------------------------------------------------
# 基础模型预训练（Base Pre-training）
# -----------------------------------------------------------------------------

# d20模型有561M参数。
# Chinchilla定律：#tokens = 20X #params，所以我们需要 561e6 * 20 = 11.2B tokens。
# 假设我们的分词器是4.8 chars/token，这是 11.2B * 4.8 ≈ 54B chars。
# 在250M chars/shard下，这是 54B / 250M ≈ 216个分片用于预训练。
# 为安全起见向上取整到240。每个分片约100MB，下载约24GB数据到磁盘。
# （整个数据集可用的分片总数是1822个。）
echo "等待数据集下载完成..."
# ├─ echo：输出文本到标准输出
# └─ 作用：友好提示，告知用户正在等待

wait $DATASET_DOWNLOAD_PID
# ├─ wait：等待指定进程完成
# └─ $DATASET_DOWNLOAD_PID：前面保存的后台下载进程的PID
#     作用：确保240个分片全部下载完成后再开始训练

# 使用的进程数/GPU数
NPROC_PER_NODE=8
# ├─ 变量定义：每个节点的进程数（对应GPU数）
# └─ 说明：8XH100意味着8个GPU，每个GPU一个进程

# 预训练d20模型（depth=20，561M参数）
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
# ├─ torchrun：PyTorch分布式训练启动器
# ├─ --standalone：单节点模式（不需要多节点协调）
# │   说明：所有进程都在同一台机器上
# ├─ --nproc_per_node=$NPROC_PER_NODE：每个节点的进程数（8）
# │   说明：启动8个进程，每个进程管理一个GPU
# ├─ -m scripts.base_train：以模块方式运行训练脚本
# ├─ --：分隔torchrun参数和脚本参数
# └─ 脚本参数：
#     ├─ --depth=20：模型深度（层数通过公式计算：num_layers = depth + 12）
#     │   d20模型：32层，561M参数
#     └─ --run=$WANDB_RUN：WandB运行名称（dummy或自定义）

# 在更大块的训练/验证数据上评估模型并抽取一些样本
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# ├─ torchrun：分布式运行（评估也要用多GPU加速）
# └─ scripts.base_loss：评估脚本
#     作用：
#     - 在更多数据（约2.5B tokens）上计算训练集和验证集的BPB
#     - 生成文本样本展示模型能力
#     - 结果记录到report

# 在CORE任务上评估模型（推理基准测试）
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
# ├─ scripts.base_eval：CORE评估脚本
# └─ 作用：
#     - 评估多个推理任务：ARC、HellaSwag、MMLU、TruthfulQA等
#     - 计算CORE分数（综合推理能力指标）
#     - 结果记录到report和CSV文件

# -----------------------------------------------------------------------------
# 中期训练（Midtraining）- 教模型对话特殊token、工具使用、多选题
# -----------------------------------------------------------------------------

# 下载2.3MB的合成身份对话，为nanochat赋予个性
# 详见dev/gen_sft_data.py了解数据准备细节，以及如何轻松调整它
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# ├─ curl：下载工具
# ├─ -L：跟随重定向
# ├─ -o：output（指定输出文件路径）
# │   $NANOCHAT_BASE_DIR/identity_conversations.jsonl
# └─ https://...：S3存储桶URL
#     说明：下载包含约100个合成对话的JSONL文件（2.3MB）
#           这些对话定义了nanochat的"身份"和回答风格

# 运行中期训练并评估模型
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
# ├─ scripts.mid_train：中期训练脚本
# └─ 作用：
#     - 继续训练base模型，教它学会：
#       * 对话格式（<|user_start|>, <|assistant_start|>等特殊token）
#       * 工具使用（Python计算器）
#       * 多选题格式
#       * SpellingBee和SimpleSpelling任务
#       * Identity对话（个性化回答）
#     - 训练约1-2个epoch
#     - 学习率较低（继续训练而非重新学习）

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
# ├─ scripts.chat_eval：聊天模型评估脚本
# ├─ --：分隔torchrun和脚本参数
# ├─ -i mid：input model tag，指定评估mid模型
# └─ 作用：
#     - 评估多个对话任务：GSM8K、HumanEval、MMLU、ARC
#     - 计算ChatCORE分数
#     - 生成评估报告

# -----------------------------------------------------------------------------
# 监督微调（SFT）- 领域适配，每行独立处理
# -----------------------------------------------------------------------------

# 训练SFT并立即重新评估（应该会看到小幅提升）
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
# ├─ scripts.chat_sft：监督微调脚本
# └─ 作用：
#     - 在混合任务数据集上微调模型：
#       * ARC-Easy/Challenge（推理）
#       * GSM8K（数学）
#       * SmolTalk（通用对话）
#       * CustomJSON（身份对话）
#       * SpellingBee（拼写计数）
#     - 训练1-2个epoch
#     - 每个任务独立优化，提升特定能力

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
# ├─ -i sft：评估sft模型（监督微调后的模型）
# └─ 预期：相比mid模型，任务性能应有小幅提升（1-3%）

# 通过CLI与模型聊天！去掉-p参数可以交互式聊天
# python -m scripts.chat_cli -p "Why is the sky blue?"
# ├─ scripts.chat_cli：命令行聊天界面
# ├─ -p：prompt（提示词），提供问题进行单次问答
# └─ 不带-p：进入交互模式，可以多轮对话
#     特殊命令：'quit'/'exit'退出，'clear'清空对话历史

# 更好的方式，通过漂亮的WebUI（ChatGPT风格）与模型聊天
# python -m scripts.chat_web
# ├─ scripts.chat_web：Web界面聊天服务器
# └─ 作用：
#     - 启动FastAPI服务器（默认端口8000）
#     - 提供ChatGPT风格的Web界面
#     - 支持流式输出（实时显示生成的文本）
#     - 支持多GPU并行推理（worker pool）
#     - 访问：http://localhost:8000

# -----------------------------------------------------------------------------
# 强化学习（RL）- 可选，目前仅针对GSM8K
# （已注释，可选步骤）
# -----------------------------------------------------------------------------

# 运行强化学习
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# ├─ scripts.chat_rl：强化学习训练脚本
# └─ 作用：
#     - 使用简化的GRPO/REINFORCE算法
#     - 在GSM8K数学问题上训练
#     - 通过Pass@k奖励优化模型
#     - token级别的奖励归一化
#     - 预期：GSM8K准确率提升5-10%

# 仅在GSM8K上评估RL模型
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K
# ├─ -i rl：评估rl模型（强化学习后的模型）
# └─ -a GSM8K：all tasks，仅评估GSM8K任务
#     说明：-a参数指定要评估的特定任务列表

# -----------------------------------------------------------------------------
# 生成完整报告
# -----------------------------------------------------------------------------
# 通过组合所有部分生成完整报告
# report.md是输出文件，为方便起见会复制到当前目录
python -m nanochat.report generate
# ├─ nanochat.report generate：生成最终报告
# └─ 作用：
#     - 从$NANOCHAT_BASE_DIR/report/收集所有部分
#     - 组合成完整的Markdown报告
#     - 包含：系统信息、训练指标、评估结果、样本输出
#     - 输出到两个位置：
#       * $NANOCHAT_BASE_DIR/report/report.md（原始位置）
#       * ./report.md（当前目录，方便查看）
