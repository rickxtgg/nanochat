"""
基础模型 CORE 指标评估脚本

功能说明：
本脚本用于评估训练好的基础模型在 CORE 基准测试上的表现。
CORE (Comprehensive Reasoning Evaluation) 是一个综合性基准测试集，
评估模型在多种推理任务上的能力。

运行方式：

1. 单GPU评估：
   python -m scripts.base_eval

2. 多GPU分布式评估（推荐）：
   torchrun --nproc_per_node=8 -m scripts.base_eval
   说明：使用多GPU可以加速评估过程

3. 评估 HuggingFace 模型：
   python -m scripts.base_eval --hf-path openai-community/gpt2

4. 限制评估样本数（调试用）：
   python -m scripts.base_eval --max-per-task 50

评估指标说明：
- CORE metric: 综合推理能力评分（0-1范围，越高越好）
- 基于多个任务的性能，使用随机基线进行归一化
- 输出结果包含每个任务的准确率和中心化得分

输出：
- 控制台：打印所有任务的详细评估结果
- CSV文件：保存在 base_eval/ 目录下，便于后续分析
- 实验报告：记录到项目报告系统
"""
import os  # 文件和目录操作
import csv  # CSV文件读写
import time  # 时间测量
import json  # JSON数据处理
import yaml  # YAML配置文件读取
import shutil  # 高级文件操作
import random  # 随机数生成
import zipfile  # ZIP文件解压
import tempfile  # 临时目录管理
from contextlib import nullcontext  # 空上下文管理器

import torch  # PyTorch深度学习框架

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock  # 通用工具函数
from nanochat.tokenizer import HuggingFaceTokenizer  # HuggingFace分词器
from nanochat.checkpoint_manager import load_model  # 模型检查点加载
from nanochat.core_eval import evaluate_task  # 核心评估任务

# =============================================================================
# nanochat 专用函数：处理 I/O 和数据准备
# =============================================================================

# CORE 基准测试数据包的下载URL（约162MB）
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    """
    解压评估数据包并放置到项目目录
    
    功能：
    - 将下载的 eval_bundle.zip 解压到临时目录
    - 移动解压后的内容到项目基础目录
    - 这是下载后处理的回调函数
    
    参数：
        file_path: 下载的 ZIP 文件路径
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        # 解压到临时目录
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        # 移动到目标位置
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    在 CORE 基准测试上评估基础模型
    
    功能：
    - 下载并准备评估数据（如果尚未准备）
    - 遍历所有评估任务并计算准确率
    - 使用随机基线进行归一化，得到中心化得分
    - 计算综合 CORE 指标
    
    参数：
        model: 要评估的模型
        tokenizer: 对应的分词器
        device: 运行设备（cuda/cpu/mps）
        max_per_task: 每个任务最多评估的样本数（-1 表示使用全部数据）
        
    返回：
        包含评估结果的字典：
        - results: 原始准确率（按任务）
        - centered_results: 中心化得分（按任务）
        - core_metric: 综合 CORE 指标
    """
    # ============= 加载配置和任务元数据 =============
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # 下载评估数据包（如果需要）
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    
    # 读取任务配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # ============= 加载随机基线值 =============
    # 用于归一化评估结果
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # ============= 遍历评估每个任务 =============
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        # 加载任务数据
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # 打乱数据：许多数据集是有序的，但我们希望支持使用子集进行调试
        shuffle_rng = random.Random(1337)  # 使用固定种子保证可复现
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # 运行任务评估
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        # 计算中心化得分：将准确率相对于随机基线归一化到 [0, 1] 范围
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    # ============= 计算综合 CORE 指标 =============
    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# =============================================================================
# HuggingFace 模型加载工具和模型包装器
# =============================================================================

class ModelWrapper:
    """
    HuggingFace 模型的轻量级包装器
    
    目的：统一 nanochat 模型和 HuggingFace 模型的接口
    使得评估代码可以透明地处理两种类型的模型
    """
    def __init__(self, model, max_seq_len=None):
        """
        初始化模型包装器
        
        参数：
            model: HuggingFace 模型实例
            max_seq_len: 最大序列长度限制（可选）
        """
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        """
        模型前向传播
        
        参数：
            input_ids: 输入token IDs张量
            
        返回：
            logits: 模型输出的对数概率
        """
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits

def load_hf_model(hf_path: str, device):
    """
    从 HuggingFace Hub 加载模型
    
    功能：
    - 下载并加载 HuggingFace 预训练模型
    - 将模型移动到指定设备
    - 加载对应的分词器
    - 包装模型以统一接口
    
    参数：
        hf_path: HuggingFace 模型路径（例如 "openai-community/gpt2"）
        device: 目标设备（cuda/cpu/mps）
        
    返回：
        (model, tokenizer): 包装后的模型和分词器
    """
    print0(f"Loading model from: {hf_path}")
    # 加载因果语言模型
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    # 特定模型的序列长度限制（GPT-2为1024）
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    # 加载分词器
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer

# =============================================================================
# 主程序
# =============================================================================
def main():
    """主评估流程"""
    import argparse
    parser = argparse.ArgumentParser(description="评估基础模型的 CORE 指标")
    parser.add_argument('--hf-path', type=str, default=None, help='要评估的 HuggingFace 模型路径（例如 openai-community/gpt2）')
    parser.add_argument('--max-per-task', type=int, default=-1, help='每个任务最多评估的样本数（-1 = 使用全部数据）')
    args = parser.parse_args()

    # ============= 分布式训练和精度设置 =============
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # ============= 加载模型和分词器 =============
    if args.hf_path is not None:
        # 从 HuggingFace 加载模型
        hf_path = args.hf_path
        print0(f"Loading huggingface model from: {hf_path}")
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path  # 用于日志记录
        model_slug = hf_path.replace("/", "-")  # 用于输出文件名
    else:
        # 从本地文件系统加载 nanochat 模型
        model, tokenizer, meta = load_model("base", device, phase="eval")
        model_name = f"base_model (step {meta['step']})"  # 用于日志记录
        model_slug = f"base_model_{meta['step']:06d}"  # 用于输出文件名

    # ============= 运行评估 =============
    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)

    # ============= 写入结果到CSV文件（仅主进程） =============
    core_metric = None
    centered_results = {}
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        
        # 写入CSV文件
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        
        # 同时打印到控制台
        print0("="*80)
        print0(f"Model: {model_name}")
        print0("="*80)
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            print0(f.read())

    # ============= 记录到实验报告 =============
    from nanochat.report import get_report
    get_report().log(section="Base model evaluation", data=[
        {
            "Model": model_name,
            "CORE metric": core_metric,
        },
        centered_results,  # 完整的任务得分表
    ])

    # ============= 清理资源 =============
    compute_cleanup()

if __name__ == "__main__":
    main()
