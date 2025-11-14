"""
CORE评估指标实现

CORE（Comprehensive Reasoning Evaluation）是DCLM论文中描述的综合推理评估指标。

论文链接：
    https://arxiv.org/abs/2406.11794

支持的任务类型：
    - multiple_choice: 多项选择任务
    - schema: Schema任务（上下文不同，延续相同）
    - language_modeling: 语言建模任务

评估方法：
    - 多项选择：选择平均损失最低的选项
    - Schema：选择平均损失最低的上下文
    - 语言建模：检查是否能正确预测所有延续token

已知问题：
    - 除了SQuAD外，所有任务都基本匹配。我们得到31%，参考值是37%。需要找出原因。
"""
import random

from jinja2 import Template
import torch
import torch.distributed as dist

# =============================================================================
# 提示词渲染工具
# =============================================================================

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """
    为多项选择问题渲染完整提示词
    
    参数：
        item: 题目字典，包含'query'和'choices'字段
        continuation_delimiter: 延续分隔符（如' '或'\n'）
        fewshot_examples: 少样本示例列表（可选）
    
    返回：
        prompts: 为每个选项渲染的提示词列表
    
    提示词格式：
        [少样本示例1问题][分隔符][少样本示例1正确答案]
        [少样本示例2问题][分隔符][少样本示例2正确答案]
        ...
        [当前问题][分隔符][当前选项]
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """
    为Schema问题渲染完整提示词
    
    参数：
        item: 题目字典，包含'context_options'和'continuation'字段
        continuation_delimiter: 延续分隔符
        fewshot_examples: 少样本示例列表（可选）
    
    返回：
        prompts: 为每个上下文选项渲染的提示词列表
    
    Schema任务特点：
        - 上下文（context）不同
        - 延续（continuation）相同
        - 选择能产生最低损失的上下文
    
    提示词格式：
        [少样本示例1正确上下文][分隔符][少样本示例1延续]
        [少样本示例2正确上下文][分隔符][少样本示例2延续]
        ...
        [当前上下文选项][分隔符][当前延续]
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    为语言建模任务渲染完整提示词
    
    参数：
        item: 题目字典，包含'context'和'continuation'字段
        continuation_delimiter: 延续分隔符
        fewshot_examples: 少样本示例列表（可选）
    
    返回：
        [prompt_without, prompt_with]: 两个提示词
            - prompt_without: 不包含延续的提示词
            - prompt_with: 包含延续的提示词
    
    注意：
        我们在模板中手动修剪上下文，因为某些数据集似乎有尾随空格（我们不需要）。
        这对于正确的token前缀检测很重要。
    
    提示词格式：
        [少样本示例1上下文][分隔符][少样本示例1延续]
        [少样本示例2上下文][分隔符][少样本示例2延续]
        ...
        [当前上下文][分隔符][当前延续]（可选）
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # 返回两个提示词：不包含延续和包含延续
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # 由于数据存储方式，我认为在LM情况下需要在这里strip。
    # 否则prompt_without可能会有尾随空格（会被吸收到prompt_with的下一个token中），
    # 这意味着我们无法在token空间中获得干净的前缀来检测最终的延续。分词器的问题...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


# =============================================================================
# Token序列处理工具
# =============================================================================

def find_common_length(token_sequences, direction='left'):
    """
    查找token序列间的公共前缀或后缀长度
    
    参数：
        token_sequences: token序列列表
        direction: 'left'表示前缀，'right'表示后缀
    
    返回：
        公共部分的长度
    
    用途：
        - 在多项选择中找到公共上下文的结尾（前缀）
        - 在Schema任务中找到公共延续的开始（后缀）
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # 找到token序列开始不同的第一个位置
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """
    堆叠token序列列表，右侧填充到最长长度
    
    参数：
        tokens: token序列列表（每个序列长度可能不同）
        pad_token_id: 填充token的ID
    
    返回：
        input_ids: 形状为(batch_size, max_seq_len)的张量
    """
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    """
    批处理多项选择任务的序列
    
    特点：
        - 上下文相同，延续不同（公共前缀）
        - 需要找到每个选项延续的起始和结束位置
    
    参数：
        tokenizer: 分词器实例
        prompts: 提示词列表（每个选项一个）
    
    返回：
        (tokens, start_indices, end_indices): 元组
            - tokens: token序列列表
            - start_indices: 每个延续的起始索引
            - end_indices: 每个延续的结束索引
    """
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # 找出每个延续的起始和结束位置
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    """
    批处理Schema任务的序列
    
    特点：
        - 上下文不同，延续相同（公共后缀）
        - 需要找到每个上下文的起始和结束位置
    
    参数：
        tokenizer: 分词器实例
        prompts: 提示词列表（每个上下文选项一个）
    
    返回：
        (tokens, start_indices, end_indices): 元组
            - tokens: token序列列表
            - start_indices: 每个上下文的起始索引
            - end_indices: 每个上下文的结束索引
    """
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # 找出每个上下文的起始和结束位置
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    """
    批处理语言建模任务的序列
    
    特点：
        - 有两个提示词：不包含延续和包含延续
        - 只需要包含延续的提示词（batch size为1）
    
    参数：
        tokenizer: 分词器实例
        prompts: [prompt_without, prompt_with]
    
    返回：
        (tokens, start_indices, end_indices): 元组
            - tokens: 包含延续的token序列（列表形式，长度为1）
            - start_indices: 延续的起始索引（列表形式）
            - end_indices: 延续的结束索引（列表形式）
    """
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # 在LM任务中我们只需要包含延续的提示词，即batch size为1
    return [tokens_with], [start_idx], [end_idx]


# =============================================================================
# 模型评估函数
# =============================================================================

@torch.no_grad()
def forward_model(model, input_ids):
    """
    模型前向传播，计算损失和预测
    
    参数：
        model: 语言模型
        input_ids: 形状为(B, T)的token ID张量
    
    返回：
        (losses, predictions): 元组
            - losses: 形状为(B, T)的损失张量（最后一列为nan）
            - predictions: 形状为(B, T)的argmax预测张量
    
    实现细节：
        - 使用torch.roll获取自回归目标（将输入向左移动一位）
        - 计算所有位置的交叉熵损失
        - 最后一列设为nan，因为没有自回归目标
        - 返回每个位置的argmax预测
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # 将张量向左滚动一位以获取（自回归）目标ID
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # 计算所有位置的交叉熵
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # 将最后一列设为nan，因为那里没有自回归损失
    losses[:, -1] = float('nan')
    # 获取每个位置的argmax预测
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """
    评估单个示例
    
    参数：
        idx: 示例索引
        model: 语言模型
        tokenizer: 分词器
        data: 数据集
        device: 设备
        task_meta: 任务元数据，包含：
            - task_type: 任务类型（'multiple_choice'/'schema'/'language_modeling'）
            - num_fewshot: 少样本示例数量
            - continuation_delimiter: 延续分隔符
    
    返回：
        is_correct: 布尔值，表示答案是否正确
    
    评估流程：
        1. 采样少样本示例（排除当前项）
        2. 根据任务类型渲染提示词并批处理序列
        3. 如果需要，截断序列以适应模型max_seq_len
        4. 堆叠序列为批次并移至设备
        5. 前向传播模型获取损失和预测
        6. 根据任务类型判断正确性：
           - LM：检查是否所有预测token都正确
           - MC/Schema：选择平均损失最低的选项
    """
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # 采样少样本示例（排除当前项）
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # 根据任务类型渲染提示词并批处理序列
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # 某些模型无法前向传播超过特定长度的序列（例如GPT-2）
    # 在这些情况下，我们必须将序列截断到最大长度并调整索引
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])  # 取最后max_tokens个token
                new_start_idxs.append(s - num_to_crop)  # 向下移动索引
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t)  # 保持不变
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # 将所有序列堆叠成一个批次
    pad_token_id = tokenizer.get_bos_token_id()  # 使用BOS作为pad token是可以的
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # 前向传播模型，获取每个token的自回归损失和argmax预测
    losses, predictions = forward_model(model, input_ids)

    # 检查损失/预测是否正确
    if task_type == 'language_modeling':
        # 语言建模任务当前总是batch size为1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i]自回归预测input_ids[i+1]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        # 对于MC/schema：找到平均损失最低的选项
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    评估一个任务的所有示例
    
    参数：
        model: 语言模型
        tokenizer: 分词器
        data: 数据集
        device: 设备
        task_meta: 任务元数据
    
    返回：
        mean_correct: 平均正确率（0-1之间的浮点数）
    
    分布式处理：
        - 如果使用torchrun运行，自动将示例分配到所有进程
        - 每个rank处理一部分示例（按world_size间隔）
        - 通过all_reduce同步所有进程的结果
        - 最后计算总体平均正确率
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    
    # 将示例分配到每个rank（间隔为world_size）
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    
    # 如果运行分布式，在所有进程间同步结果
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    
    # 计算平均值
    mean_correct = correct.mean().item()
    return mean_correct
