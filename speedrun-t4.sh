#!/bin/bash

# ==============================================================================
# 通用版 Nanochat Speedrun 脚本 (扁平路径版)
# ==============================================================================

# 1. 锁定当前脚本所在目录为工作根目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 2. === 路径配置 (修改版) ===
# 按照要求：NANOCHAT_BASE_DIR 直接指向脚本所在目录
# 其他缓存目录也不再放入 data/ 子目录，而是直接放在根目录下
export NANOCHAT_BASE_DIR="$SCRIPT_DIR"

# 日志目录
LOG_DIR="$SCRIPT_DIR/logs"

# 创建必要的目录
mkdir -p "$LOG_DIR"
# 注意：NANOCHAT_BASE_DIR 就是当前目录，无需创建，但为了保险还是 mkdir 一下
mkdir -p "$NANOCHAT_BASE_DIR"

# === 日志文件设置 ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/speedrun_${TIMESTAMP}.log"

# 定义日志函数
log_info() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log_section() {
  echo ""
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ========================================"
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ========================================"
}

# 重定向所有输出 (标准输出 + 错误输出) 到 屏幕 + 日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

# 3. === 环境变量设置 ===
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# 设置缓存路径 (去掉了 /data/ 层级)
export HF_HOME="$SCRIPT_DIR/huggingface_cache"
export TORCH_HOME="$SCRIPT_DIR/torch_cache"
mkdir -p "$HF_HOME"
mkdir -p "$TORCH_HOME"

# 4. === 启动信息 ===
echo "=== NanoChat 训练启动 (Path Modified) ==="
echo "启动时间: $(date)"
echo "脚本路径: $SCRIPT_DIR"
echo "基准路径: $NANOCHAT_BASE_DIR"
echo "日志文件: $LOG_FILE"
echo "=================================="

# 5. === 硬件检测 ===
GPU_COUNT=$(nvidia-smi -L | wc -l)
log_info "检测到 GPU 数量: $GPU_COUNT"

# -----------------------------------------------------------------------------
# 阶段 1: 环境依赖
# -----------------------------------------------------------------------------
log_section "阶段 1/8: Python 环境依赖安装"

log_info "正在升级 pip..."
python3 -m pip install --upgrade pip

log_info "安装项目依赖 (包含 GPU 支持)..."
python3 -m pip install -e ".[gpu]"

log_info "安装编译工具 maturin..."
python3 -m pip install maturin

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
    log_info "WandB 未配置，使用 dummy 模式"
else
    log_info "WandB 已启用，Run ID: $WANDB_RUN"
fi

python3 -m nanochat.report reset

# -----------------------------------------------------------------------------
# 阶段 2: Rust 环境与 Tokenizer (修复版)
# -----------------------------------------------------------------------------
log_section "阶段 2/8: Rust 环境配置与编译"

# 设置 Rust 路径 (去掉了 /data/ 层级)
export RUSTUP_HOME="$SCRIPT_DIR/rustup"
export CARGO_HOME="$SCRIPT_DIR/cargo"
export PATH="$CARGO_HOME/bin:$PATH"

# 1. 确保 rustup 存在
if ! command -v rustup &> /dev/null; then
    log_info "未检测到 rustup，正在下载安装..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
else
    log_info "检测到 rustup 已存在"
fi

# 2. [关键修复] 显式设置 stable 工具链
log_info "正在强制配置 stable 工具链..."
rustup default stable
rustup update stable

# 3. 验证
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust 仍然无法使用，请尝试手动运行: export PATH=\$PATH:$CARGO_HOME/bin && rustup default stable"
    exit 1
else
    log_info "Rust 环境就绪: $(rustc --version)"
fi

log_info "正在编译 Tokenizer (Release 模式)..."
maturin develop --release --manifest-path rustbpe/Cargo.toml

# -----------------------------------------------------------------------------
# 阶段 3: 数据准备
# -----------------------------------------------------------------------------
log_section "阶段 3/8: 数据下载与处理"

log_info "下载前 8 个数据分片..."
python3 -m nanochat.dataset -n 8

log_info "后台启动剩余数据下载 (n=240)..."
python3 -m nanochat.dataset -n 48 &
DATASET_DOWNLOAD_PID=$!
log_info "后台下载进程 PID: $DATASET_DOWNLOAD_PID"

log_info "开始训练 Tokenizer (词表构建)..."
python3 -m scripts.tok_train --max_chars=2000000000

log_info "评估 Tokenizer..."
python3 -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 阶段 4: 预训练 (Pretraining)
# -----------------------------------------------------------------------------
log_section "阶段 4/8: Base Model 预训练"

log_info "等待数据下载完成..."
wait $DATASET_DOWNLOAD_PID
log_info "数据下载完毕，开始训练循环。"

log_info "启动 base_train (GPU=$GPU_COUNT)..."
# 如果 AutoDL 显存不足，请添加 --device_batch_size=4
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_train -- --depth=4 --device_batch_size=4 --run=$WANDB_RUN

log_info "计算 Base Loss..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_loss

log_info "评估 Base Model (CORE tasks)..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_eval

# -----------------------------------------------------------------------------
# 阶段 5: Mid-training
# -----------------------------------------------------------------------------
log_section "阶段 5/8: Mid-training"

log_info "下载对话数据集..."
# 文件将直接下载到 $SCRIPT_DIR/identity_conversations.jsonl
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

log_info "运行 Mid-training..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.mid_train --device_batch_size=4 -- --run=$WANDB_RUN

log_info "评估 Mid 模型..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# 阶段 6: SFT
# -----------------------------------------------------------------------------
log_section "阶段 6/8: SFT 监督微调"

log_info "运行 SFT..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.chat_sft --device_batch_size=2 -- --run=$WANDB_RUN

log_info "评估 SFT 模型..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# 阶段 8: 报告
# -----------------------------------------------------------------------------
log_section "阶段 8/8: 任务完成"

log_info "生成最终 markdown 报告..."
python3 -m nanochat.report generate

log_section "训练结束 SUMMARY"
echo "日志文件: $LOG_FILE"
echo "所有数据已保存在当前目录下。"
echo "启动 Web 界面命令: python3 -m scripts.chat_web"
echo "========================================"