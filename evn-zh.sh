#!/bin/bash

# ==============================================================================
# 国内专用版 Nanochat Speedrun 脚本 (Rust Fix + TUNA镜像)
# ==============================================================================

# 1. 锁定当前脚本所在目录为工作根目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 2. === 路径配置 ===
export NANOCHAT_BASE_DIR="$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"
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

# 重定向所有输出
exec > >(tee -a "$LOG_FILE") 2>&1

# 3. === 环境变量设置 ===
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

export HF_HOME="$SCRIPT_DIR/huggingface_cache"
export TORCH_HOME="$SCRIPT_DIR/torch_cache"
mkdir -p "$HF_HOME"
mkdir -p "$TORCH_HOME"

# 4. === 启动信息 ===
echo "=== NanoChat 训练启动 (Clean Rust Install) ==="
echo "启动时间: $(date)"
echo "工作目录: $SCRIPT_DIR"
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
python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

log_info "安装项目依赖..."
python3 -m pip install -e ".[gpu]" -i https://pypi.tuna.tsinghua.edu.cn/simple

log_info "安装 maturin..."
python3 -m pip install maturin -i https://pypi.tuna.tsinghua.edu.cn/simple

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
    log_info "WandB 未配置，使用 dummy 模式"
else
    log_info "WandB 已启用，Run ID: $WANDB_RUN"
fi

python3 -m nanochat.report reset

# -----------------------------------------------------------------------------
# 阶段 2: Rust 环境与 Tokenizer (强制重装版)
# -----------------------------------------------------------------------------
log_section "阶段 2/8: Rust 环境配置与编译"

export RUSTUP_HOME="$SCRIPT_DIR/rustup"
export CARGO_HOME="$SCRIPT_DIR/cargo"
export PATH="$CARGO_HOME/bin:$PATH"

# === [关键步骤] 强制清理旧的 Rust 环境 ===
# 解决 "no release found" 和脏数据问题
if [ -d "$RUSTUP_HOME" ] || [ -d "$CARGO_HOME" ]; then
    log_info "⚠️  检测到残留的 Rust 环境，正在执行清理以确保安装成功..."
    rm -rf "$RUSTUP_HOME" "$CARGO_HOME"
fi

# === [加速配置] 使用清华源加速 Rustup ===
export RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup"
export RUSTUP_UPDATE_ROOT="https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"
log_info "已配置清华镜像: $RUSTUP_DIST_SERVER"

# 1. 安装 Rustup
log_info "正在下载并安装 Rustup..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path

# 2. 安装 Stable 工具链
log_info "正在安装 stable 工具链..."
rustup default stable

# 3. === [加速配置] 配置 Cargo 镜像 (Rsproxy Sparse) ===
# 依赖包下载继续用 Rsproxy，因为它支持稀疏索引，比清华全量索引更快
log_info "配置 Cargo 镜像源..."
mkdir -p "$CARGO_HOME"
cat > "$CARGO_HOME/config.toml" <<EOF
[source.crates-io]
replace-with = 'rsproxy-sparse'
[source.rsproxy]
registry = "https://rsproxy.cn/crates.io-index"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
[registries.rsproxy]
index = "https://rsproxy.cn/crates.io-index"
[net]
git-fetch-with-cli = true
EOF

# 验证
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust 安装失败。请检查网络连接。"
    exit 1
else
    log_info "✅ Rust 环境就绪: $(rustc --version)"
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
python3 -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
log_info "后台下载进程 PID: $DATASET_DOWNLOAD_PID"

log_info "开始训练 Tokenizer..."
python3 -m scripts.tok_train --max_chars=2000000000

log_info "评估 Tokenizer..."
python3 -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 阶段 4: 预训练 (Pretraining)
# -----------------------------------------------------------------------------
log_section "阶段 4/8: Base Model 预训练"

log_info "等待数据下载完成..."
wait $DATASET_DOWNLOAD_PID
log_info "数据下载完毕。"

log_info "启动 base_train (GPU=$GPU_COUNT)..."
# 注意：如果 AutoDL 显存 OOM，请取消下一行注释，并注释掉再下一行
# torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_train -- --depth=20 --device_batch_size=4 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_train -- --depth=20 --run=$WANDB_RUN

log_info "计算 Base Loss..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_loss

log_info "评估 Base Model..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_eval

# -----------------------------------------------------------------------------
# 阶段 5: Mid-training
# -----------------------------------------------------------------------------
log_section "阶段 5/8: Mid-training"

log_info "下载对话数据集..."
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

log_info "运行 Mid-training..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.mid_train -- --run=$WANDB_RUN

log_info "评估 Mid 模型..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# 阶段 6: SFT
# -----------------------------------------------------------------------------
log_section "阶段 6/8: SFT 监督微调"

log_info "运行 SFT..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.chat_sft -- --run=$WANDB_RUN

log_info "评估 SFT 模型..."
torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# 阶段 8: 报告
# -----------------------------------------------------------------------------
log_section "阶段 8/8: 任务完成"

python3 -m nanochat.report generate

log_section "训练结束 SUMMARY"
echo "日志文件: $LOG_FILE"
echo "启动 Web 界面: python3 -m scripts.chat_web"
echo "========================================"