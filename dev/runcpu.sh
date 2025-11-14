#!/bin/bash

# ============================================
# nanochat CPU 演示运行脚本
# ============================================
#
# 功能说明：
# 本脚本演示如何在CPU（或Macbook的MPS）上运行 nanochat 项目的主要代码路径。
# 
# 运行方法：
# bash dev/runcpu.sh
#
# 重要提示：
# - 训练大语言模型需要GPU计算资源和相当的资金投入（$$$）
# - 在个人电脑（如Macbook）上运行无法获得实用的训练效果
# - 本脚本仅作为教育/演示用途，不应期望获得良好的模型性能
# - 这也是为什么将此脚本放在 dev/ 目录中的原因
# 
# 用途：
# - 快速验证代码逻辑是否正确
# - 学习项目的工作流程
# - 在没有GPU的情况下进行功能测试
# ============================================

# ============================================
# 环境设置和初始化
# ============================================

# 设置 OpenMP 线程数为 1，避免 CPU 过度使用导致的性能问题
export OMP_NUM_THREADS=1

# 设置项目基础目录，用于存储缓存、模型检查点等
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# 检查 uv 包管理器是否已安装，如果没有则自动安装
# uv 是一个现代化的 Python 包管理器，速度更快
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境（如果不存在）
[ -d ".venv" ] || uv venv

# 同步安装依赖，使用 CPU 版本的额外依赖
uv sync --extra cpu

# 激活虚拟环境
source .venv/bin/activate

# 设置 Weights & Biases (WandB) 运行名称
# 如果未设置，则使用 "dummy" 作为默认值（演示模式，不上传实验数据）
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# 安装 Rust 工具链（如果尚未安装）
# Rust 用于编译高性能的 BPE 分词器
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 编译并安装 Rust BPE 分词器模块（release 模式以获得最佳性能）
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# ============================================
# 第一步：准备工作 - 重置报告
# ============================================
# 清空之前的训练报告，开始新的实验记录
python -m nanochat.report reset

# ============================================
# 第二步：训练分词器 (Tokenizer)
# ============================================
# 分词器训练在约 10亿字符的数据上进行

# 准备数据集（4个数据分片）
python -m nanochat.dataset -n 4

# 训练分词器，最大处理 10亿字符
python -m scripts.tok_train --max_chars=1000000000

# 评估分词器性能（压缩率、词汇覆盖率等）
python -m scripts.tok_eval

# ============================================
# 第三步：基础预训练 (Base Pre-training)
# ============================================
# 在CPU上训练一个非常小的4层模型
# 注意：这只是演示代码路径，训练效果会很差

# 训练参数说明：
# --depth=4: 模型深度为4层（非常小的模型，适合CPU演示）
# --max_seq_len=1024: 最大序列长度为1024个token
# --device_batch_size=1: 每个设备的批次大小为1（CPU资源有限）
# --total_batch_size=1024: 总批次大小为1024个token（通过梯度累积实现）
# --eval_every=50: 每50步评估一次
# --eval_tokens=4096: 评估时使用4096个token
# --core_metric_every=50: 每50步计算核心指标
# --core_metric_max_per_task=12: 每个任务最多评估12个样本
# --sample_every=50: 每50步生成一次文本样本
# --num_iterations=50: 总共运行50步优化（增大此值可获得更好结果，但耗时更长）
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --core_metric_every=50 \
    --core_metric_max_per_task=12 \
    --sample_every=50 \
    --num_iterations=50

# 评估基础模型的损失
python -m scripts.base_loss --device_batch_size=1 --split_tokens=4096

# 在标准任务上评估模型性能（每个任务最多16个样本）
python -m scripts.base_eval --max-per-task=16

# ============================================
# 第四步：中期训练 (Midtraining)
# ============================================
# 在特定领域数据上继续训练，使模型适应特定任务或领域

# 中期训练参数：
# --max_seq_len=1024: 最大序列长度
# --device_batch_size=1: 设备批次大小
# --eval_every=50: 每50步评估一次
# --eval_tokens=4096: 评估token数
# --total_batch_size=1024: 总批次大小
# --num_iterations=100: 运行100步
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --total_batch_size=1024 \
    --num_iterations=100

# 评估中期训练后的模型对话能力
# 注意：评估结果会很差，这只是为了执行代码路径
# --source=mid: 使用中期训练的检查点
# --max-new-tokens=128: 每次生成最多128个新token
# --max-problems=20: 最多评估20个问题
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# ============================================
# 第五步：监督微调 (SFT - Supervised Fine-Tuning)
# ============================================
# 使用对话数据进行监督微调，使模型学会遵循指令和进行对话

# SFT 训练参数：
# --device_batch_size=1: 设备批次大小
# --target_examples_per_step=4: 每步目标样本数（通过梯度累积实现）
# --num_iterations=100: 总迭代次数
# --eval_steps=4: 每4步评估一次
# --eval_metrics_max_problems=16: 评估时最多使用16个问题
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --num_iterations=100 \
    --eval_steps=4 \
    --eval_metrics_max_problems=16

# ============================================
# 第六步（可选）：交互式使用模型
# ============================================
# 以下命令被注释掉，可根据需要取消注释使用

# 命令行对话接口 (Chat CLI)
# 用法：通过 -p 参数提供提示词，模型会生成回答
# 示例：为什么天空是蓝色的？
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Web 界面对话 (Chat Web)
# 启动一个本地 Web 服务器，提供图形化的对话界面
# python -m scripts.chat_web

# ============================================
# 第七步：生成训练报告
# ============================================
# 汇总所有训练和评估结果，生成一份完整的实验报告
python -m nanochat.report generate
