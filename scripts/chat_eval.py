"""
对话模型评估脚本 - 多任务评估系统

功能说明：
本脚本用于评估经过SFT或RL训练的对话模型在多种任务上的表现。
通用评估逻辑在此文件中实现，任务特定的实现位于 tasks/ 目录。

支持的评估任务：
1. HumanEval: Python代码生成任务
2. MMLU: 多学科多选题测试
3. ARC-Easy/Challenge: AI2推理挑战赛
4. GSM8K: 小学数学问题
5. SpellingBee: 拼写相关任务

评估模式：
- 生成式评估（Generative）：需要模型生成完整答案（如HumanEval、GSM8K）
- 分类式评估（Categorical）：只需选择答案选项（如MMLU、ARC）

运行方式：

1. 评估单个任务（单GPU）：
   python -m scripts.chat_eval -i sft -a ARC-Easy

2. 评估单个任务（多GPU，加速）：
   torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a ARC-Easy

3. 评估所有任务：
   python -m scripts.chat_eval -i sft
   说明：不指定任务名时，将评估所有支持的任务

4. 评估多个任务：
   python -m scripts.chat_eval -i sft -a "ARC-Easy|MMLU|GSM8K"
   说明：使用 | 分隔多个任务名

5. 限制评估样本数（调试用）：
   python -m scripts.chat_eval -i sft -a GSM8K -x 100
   说明：每个任务最多评估100个问题

技术特性：
- 支持分布式评估（多GPU并行）
- 自动计算 ChatCORE 综合指标
- 支持生成式和分类式两种评估模式
- 使用随机基线进行归一化
- 结果记录到实验报告系统

输出：
- 每个任务的准确率
- ChatCORE 综合指标（如果评估了所有任务）
- 详细的进度信息
"""

import argparse  # 命令行参数解析
from functools import partial  # 偏函数工具
from contextlib import nullcontext  # 空上下文管理器

import torch  # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练工具

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type  # 通用工具函数
from nanochat.checkpoint_manager import load_model  # 模型检查点加载
from nanochat.engine import Engine  # 文本生成引擎

# 导入各个评估任务
from tasks.humaneval import HumanEval  # Python代码生成
from tasks.mmlu import MMLU  # 多学科多选题
from tasks.arc import ARC  # AI2推理挑战
from tasks.gsm8k import GSM8K  # 小学数学问题
from tasks.spellingbee import SpellingBee  # 拼写任务

# =============================================================================
# 生成式评估循环
# =============================================================================
# 适用于需要模型生成完整答案的任务（如HumanEval、GSM8K）
# 逐个问题进行采样和评估，支持pass@k评估

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):
    """
    运行生成式评估（每次一个问题，采样并评估）
    
    参数：
        task_object: 任务对象，包含问题和评估逻辑
        tokenizer: 分词器
        model: 模型实例
        engine: 生成引擎
        num_samples: 每个问题生成的样本数（用于pass@k评估）
        max_new_tokens: 每次生成的最大token数
        temperature: 生成温度
        top_k: Top-k采样参数
        max_problems: 最多评估的问题数（None表示全部）
        
    返回：
        准确率（float）
    """

    # 获取分布式训练信息
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    # 确定要评估的问题数量
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # ============= 运行评估循环 =============
    num_passed, total = 0, 0
    # 分布式评估：每个进程负责不同的问题子集
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # 将提示词token化（准备用于补全）
        encoded_prompt = tokenizer.render_for_completion(conversation)
        
        # 生成多个补全样本（用于pass@k评估）
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        
        # 解码生成的补全为文本
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        
        # 评估每个补全是否正确
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)  # pass@k: 任何一个正确即算通过

        # 统计
        total += 1
        num_passed += int(passed)

        # 在同一行更新进度（覆盖式打印）
        print(f"\r\033[K进程 {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # 完成进度行后换行
    print()

    # ============= 聚合所有进程的结果 =============
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"最终结果: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    # 返回准确率
    return num_passed/total

# =============================================================================
# 分类式评估循环
# =============================================================================
# 比生成式简单很多，因为不需要采样。可以批量处理，
# 只需检查正确答案选项的logits即可。

def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):
    """
    运行分类式评估（批量处理，检查logits）
    
    更高效的评估方式，适用于多选题任务（如MMLU、ARC）。
    不需要实际生成文本，只需比较答案选项的logits大小。
    
    参数：
        task_object: 任务对象
        tokenizer: 分词器
        model: 模型实例
        batch_size: 批次大小（可以更大，因为不需要采样）
        max_problems: 最多评估的问题数
        
    返回：
        准确率（float）
    """

    # 获取分布式信息
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()  # 使用BOS作为填充token（这些位置会被忽略）

    # ============= 批量处理独立问题 =============
    # 因为不需要采样，可以批量处理多个问题以提高效率
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)  # 向上取整除法
    num_batches = ceil_div(num_problems, batch_size)

    # ============= 运行评估 =============
    letter_to_id_cache = {}  # 缓存字母到token ID的映射（很多字母会重复出现）
    num_passed, total = 0, 0
    
    # 分布式处理：每个进程负责不同的批次
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # 准备批次数据：不同问题可能长度不同，需要填充和整理
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]  # 答案位置（最后一个token）
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]  # 填充到相同长度
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # 并行获取整个批次的logits（效率提升的关键）
        with torch.no_grad():
            logits = model(prompt_ids)  # (B, T, V)

        # ============= 基于logits选择答案 =============
        # 只关注可用答案选项对应的字母的logits
        # 注意：这大大简化了评估，因为只需比较几个字母的logits，而不需要生成完整回答
        # 更困难的替代方案是让模型生成完整回答，然后检查是否包含正确字母（如A、B、C、D）
        for idx, conversation in enumerate(conversations):
            # 获取此问题所有可用字母的token IDs
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if not letter in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "每个字母必须是单个token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            
            # 只关注答案位置和可用字母的logits
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            
            # 获取logits最大的字母（预测答案）
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            
            # 评估结果
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # ============= 聚合所有进程的结果 =============
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed/total
    print0(f"最终结果: {num_passed}/{total} ({100*average:.2f}%)")
    return average

# =============================================================================
# 任务评估统一入口
# =============================================================================

def run_chat_eval(task_name, model, tokenizer, engine,
                   batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                   max_problems=None):
    """
    运行指定任务的评估（统一入口函数）
    
    根据任务类型自动选择合适的评估方法：
    - 生成式任务：使用 run_generative_eval
    - 分类式任务：使用 run_categorical_eval
    
    参数：
        task_name: 任务名称（如 'HumanEval', 'MMLU' 等）
        model: 模型实例
        tokenizer: 分词器
        engine: 生成引擎
        batch_size: 批次大小（仅用于分类式评估）
        num_samples: 采样数（仅用于生成式评估）
        max_new_tokens: 最大生成token数（仅用于生成式评估）
        temperature: 生成温度
        top_k: Top-k采样参数
        max_problems: 最多评估的问题数
        
    返回：
        准确率（float）
    """
    # 创建任务对象
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
        'SpellingBee': partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()
    
    # 根据任务类型选择评估方法
    if task_object.eval_type == 'generative':
        # 生成式评估：需要完整生成答案
        acc = run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=max_problems)
    elif task_object.eval_type == 'categorical':
        # 分类式评估：只需比较logits
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
    else:
        raise ValueError(f"不支持的任务评估类型: {task_object.eval_type}")
    return acc

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":

    # ============= 解析命令行参数 =============
    parser = argparse.ArgumentParser(description="评估对话模型在多个任务上的表现")
    parser.add_argument('-i', '--source', type=str, required=True, help="模型来源：sft|mid|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="任务名称。默认=所有任务。使用|分隔多个任务")
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'], help="数据类型")
    parser.add_argument('-t', '--temperature', type=float, default=0.0, help="生成温度")
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512, help="最大生成token数")
    parser.add_argument('-n', '--num-samples', type=int, default=1, help="每个问题的采样数（pass@k）")
    parser.add_argument('-k', '--top-k', type=int, default=50, help="Top-k采样参数")
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='分类式评估的批次大小')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='要加载的模型标签')
    parser.add_argument('-s', '--step', type=int, default=None, help='要加载的训练步数')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='最多评估的问题数')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='设备类型：cuda|cpu|mps（空值=自动检测）')
    args = parser.parse_args()

    # ============= 初始化计算环境 =============
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # ============= 加载模型和分词器 =============
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # ============= 配置评估任务 =============
    # 所有支持的任务列表
    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    # 随机基线准确率（用于归一化）
    baseline_accuracies = {
        'ARC-Easy': 0.25,       # 四选一多选题 => 25%
        'ARC-Challenge': 0.25,  # 四选一多选题 => 25%
        'MMLU': 0.25,           # 四选一多选题 => 25%
        'GSM8K': 0.0,           # 开放式问题 => 0%
        'HumanEval': 0.0,       # 开放式问题 => 0%
        'SpellingBee': 0.0,     # 开放式问题 => 0%
    }
    # 确定要评估的任务（如果未指定则评估所有任务）
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    # ============= 顺序运行所有任务评估 =============
    results = {}
    for task_name in task_names:
        with autocast_ctx:
            acc = run_chat_eval(
                task_name,
                model, tokenizer, engine,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
            )
            results[task_name] = acc
            print0(f"{task_name} 准确率: {100 * acc:.2f}%")

    # ============= 计算 ChatCORE 综合指标 =============
    from nanochat.report import get_report
    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    
    # 计算 ChatCORE 指标（类似CORE，是中心化准确率的平均值）
    # 这样 ChatCORE 的范围从 0（随机基线）到 1（峰值性能）
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            # 中心化：(实际准确率 - 基线) / (1 - 基线)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}
    
    # ============= 记录到实验报告 =============
    get_report().log(section="Chat evaluation " + args.source, data=[
        vars(args),  # 命令行参数
        results,  # 各任务准确率
        chatcore_metric_dict,  # ChatCORE综合指标
    ])

    # ============= 清理资源 =============
    compute_cleanup()
