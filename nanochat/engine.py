"""
高效推理引擎

本模块为nanochat模型提供高效的推理引擎，专注于token序列的生成。

核心功能：
    - 接收token序列作为输入
    - 流式生成下一个token
    - 支持批量并行采样（一次预填充，多次采样）
    - KV缓存管理，减少重复计算
    - 工具使用（计算器）集成
    
设计原则：
    - 引擎只处理token ID序列，不涉及分词器（tokenization）
    - 最大化推理效率（KV缓存、批处理、流式生成）
    - 支持多样化的采样策略（temperature、top-k）
    - 状态机管理工具调用（python_start/end、output_start/end）
    
主要组件：
    - KVCache: 管理键值对缓存，支持动态扩展
    - Engine: 主推理引擎，支持流式和批量生成
    - sample_next_token: 采样函数，支持温度和top-k
    - RowState: 每个生成序列的状态跟踪
"""

import torch  # PyTorch核心库
import torch.nn.functional as F  # 神经网络函数（softmax等）
import signal  # Unix信号处理（用于超时）
import warnings  # 警告控制
from contextlib import contextmanager  # 上下文管理器装饰器
from collections import deque  # 高效双端队列（用于强制token队列）
from nanochat.common import compute_init  # 计算环境初始化
from nanochat.checkpoint_manager import load_model  # 模型加载工具

# =============================================================================
# 计算器工具辅助函数
# =============================================================================
# 这些函数支持模型的工具使用能力，允许模型执行Python表达式（如数学计算、字符串操作）
@contextmanager
def timeout(duration, formula):
    """
    超时上下文管理器
    
    参数：
        duration: 超时时长（秒）
        formula: 要执行的表达式（用于错误消息）
    
    功能：
        使用Unix信号SIGALRM在指定时间后中断执行
        防止恶意或低效的代码长时间运行
    """
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)  # 设置信号处理器
    signal.alarm(duration)  # 设置定时器
    yield
    signal.alarm(0)  # 清除定时器

def eval_with_timeout(formula, max_time=3):
    """
    带超时的安全表达式求值
    
    参数：
        formula: Python表达式字符串
        max_time: 最大执行时间（秒，默认3秒）
    
    返回：
        求值结果，失败返回None
    
    安全措施：
        - 超时限制（防止无限循环）
        - 禁用内置函数（__builtins__={}，防止危险操作）
        - 捕获所有异常（包括语法错误）
        - 忽略语法警告
    """
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                # 在空环境中求值（无内置函数，无全局变量）
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)  # 确保清除定时器
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # 忽略计算器使用错误是可以的
        return None

def use_calculator(expr):
    """
    安全地执行Python表达式（计算器功能）
    
    参数：
        expr: Python表达式字符串
    
    返回：
        求值结果，失败返回None
    
    支持的操作：
        1. 纯数学表达式：+, -, *, /, (), 数字
        2. 字符串操作：.count()方法
    
    安全限制：
        - 移除数字中的逗号（如"1,000"）
        - 禁止幂运算（**）
        - 禁止危险模式（__、import、exec等）
        - 仅允许特定字符集
        - 字符串操作仅支持.count()（可扩展）
        - 超时限制（3秒）
    """
    # 移除数字中的逗号（如"1,000" -> "1000"）
    expr = expr.replace(",", "")

    # 检查是否为纯数学表达式（旧行为）
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # 禁止幂运算符（可能导致极大数字）
            return None
        return eval_with_timeout(expr)

    # 检查是否为我们支持的字符串操作
    # 允许：字符串（单/双引号）、.count()、字母、数字、空格、括号
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # 禁止危险模式
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # 目前仅允许.count()方法（可以后续扩展）
    if '.count(' not in expr:
        return None

    # 带超时的求值
    return eval_with_timeout(expr)

# =============================================================================
# KV缓存管理
# =============================================================================
class KVCache:
    """
    键值对（Key-Value）缓存管理器
    
    功能：
        与GPT模型协同工作，维护注意力机制的KV缓存
        避免重复计算历史token的键值对，大幅提升推理速度
    
    缓存结构：
        形状：(num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        - num_layers: Transformer层数
        - 2: K（键）和V（值）
        - batch_size: 批大小（支持并行生成）
        - num_heads: 注意力头数
        - seq_len: 序列最大长度
        - head_dim: 每个注意力头的维度
    
    关键特性：
        - 懒初始化：在第一次插入时才分配内存（知道dtype/device）
        - 动态扩展：序列超出容量时自动增长（以1024为单位）
        - 位置跟踪：.pos在最后一层插入后自动前进
        - 预填充支持：从另一个缓存复制数据（用于批量采样）
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        """
        初始化KV缓存
        
        参数：
            batch_size: 批大小
            num_heads: 注意力头数
            seq_len: 序列最大长度（初始容量）
            head_dim: 每个注意力头的维度
            num_layers: Transformer层数
        """
        # 每层的K/V形状为(B, H, T, D)
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None  # 懒初始化（等待dtype/device）
        self.pos = 0  # 当前缓存位置（时间步）

    def reset(self):
        """重置位置指针（不清空缓存数据）"""
        self.pos = 0

    def get_pos(self):
        """获取当前位置指针"""
        return self.pos

    def prefill(self, other):
        """
        从另一个KV缓存预填充
        
        参数：
            other: 源KV缓存
        
        用途：
            批量采样优化：先用batch=1预填充提示词，
            然后复制到batch=N的缓存中并行生成多个样本
        
        验证规则：
            - 当前缓存必须为空（未初始化）
            - 源缓存必须非空
            - num_layers、num_heads、head_dim必须匹配
            - batch_size可以扩展（other为1，self为N）
            - seq_len：self必须>=other（容量足够）
        
        实现步骤：
            1. 验证形状兼容性
            2. 分配内存（使用other的dtype/device）
            3. 复制数据（从other到self）
            4. 更新位置指针
        """
        # 1) 验证形状
        assert self.kv_cache is None, "无法预填充非空的KV缓存"
        assert other.kv_cache is not None, "无法从空缓存预填充"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                # num_layers, K/V标记, num_heads, head_dim必须匹配
                assert dim1 == dim2, f"维度{ix}不匹配：{dim1} != {dim2}"
            elif ix == 2:
                # batch_size可以扩展（从1扩展到N）
                assert dim1 == dim2 or dim2 == 1, f"批大小不匹配：{dim1} != {dim2}"
            elif ix == 4:
                # seq_len：self必须>=other（容量足够）
                assert dim1 >= dim2, f"序列长度不匹配：{dim1} < {dim2}"
        # 2) 初始化缓存
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) 复制数据（广播batch维度，如果需要）
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) 更新位置指针
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        """
        插入新的键值对到缓存
        
        参数：
            layer_idx: Transformer层索引
            k: 键张量，形状(B, H, T_add, D)
            v: 值张量，形状(B, H, T_add, D)
        
        返回：
            key_view: 该层的完整键缓存（到当前位置），形状(B, H, t1, D)
            value_view: 该层的完整值缓存（到当前位置），形状(B, H, t1, D)
        
        关键特性：
            1. 懒初始化：第一次调用时分配内存（使用k的dtype/device）
            2. 动态扩展：容量不足时自动增长（以1024为单位，向上取整）
            3. 位置自动前进：最后一层处理完后，pos前进T_add步
            4. 返回视图：避免数据复制，直接返回缓存切片
        """
        # 懒初始化：需要知道dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        
        # 插入新键值对并返回完整缓存
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        
        # 动态扩展缓存（如果需要）
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024  # 需要的大小 + 1024缓冲
            t_needed = (t_needed + 1023) & ~1023  # 向上取整到1024的倍数
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape  # 更新形状记录
        
        # 插入k, v到缓存
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        
        # 返回该层的完整缓存（到当前位置，作为视图）
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        
        # 在Transformer的最后一层处理完后，位置指针前进
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        
        return key_view, value_view


# =============================================================================
# Token采样函数
# =============================================================================
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """
    从logits分布中采样下一个token
    
    参数：
        logits: 模型输出的logits，形状(B, vocab_size)
        rng: PyTorch随机数生成器（用于可重复性）
        temperature: 温度参数（默认1.0）
            - 0.0: 贪婪解码（argmax）
            - <1.0: 更确定性（分布更尖锐）
            - >1.0: 更随机性（分布更平滑）
        top_k: Top-K采样（可选）
            - None: 从整个词汇表采样
            - K: 只从概率最高的K个token中采样
    
    返回：
        next_token: 采样的token ID，形状(B, 1)
    
    采样策略：
        1. temperature=0.0 → 贪婪解码（确定性）
        2. top_k=None → 从所有token采样（最大随机性）
        3. top_k=K → 从top-K token采样（平衡多样性和质量）
    """
    assert temperature >= 0.0, "温度必须非负"
    
    # 温度为0：贪婪解码（选择概率最高的token）
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # Top-K采样
    if top_k is not None:
        k = min(top_k, logits.size(-1))  # k不能超过词汇表大小
        vals, idx = torch.topk(logits, k, dim=-1)  # 选出top-k个logits
        vals = vals / temperature  # 应用温度
        probs = F.softmax(vals, dim=-1)  # 归一化为概率分布
        choice = torch.multinomial(probs, num_samples=1, generator=rng)  # 采样
        return idx.gather(1, choice)  # 映射回原始token ID
    else:
        # 无Top-K限制：从整个词汇表采样
        logits = logits / temperature  # 应用温度
        probs = F.softmax(logits, dim=-1)  # 归一化为概率分布
        return torch.multinomial(probs, num_samples=1, generator=rng)  # 采样

# =============================================================================
# 生成状态跟踪
# =============================================================================

class RowState:
    """
    单个生成序列的状态跟踪器
    
    功能：
        跟踪每个生成序列（行）的状态，支持多样本并行生成
        管理工具使用状态机（计算器）
        处理强制token注入（工具输出）
    
    状态变量：
        current_tokens: 当前token序列（包括提示词和已生成部分）
        forced_tokens: 待强制注入的token队列（双端队列）
        in_python_block: 是否在python代码块内（<|python_start|>到<|python_end|>）
        python_expr_tokens: 当前python表达式的token列表
        completed: 是否已完成生成（遇到<|assistant_end|>或<|bos|>）
    
    工具使用流程：
        1. 遇到<|python_start|> → in_python_block=True
        2. 累积python表达式token到python_expr_tokens
        3. 遇到<|python_end|> → 计算表达式 → 强制注入结果token
        4. 强制token格式：<|output_start|> + 结果 + <|output_end|>
    """
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []  # 当前token序列
        self.forced_tokens = deque()  # 待强制注入的token队列
        self.in_python_block = False  # 是否在python块内
        self.python_expr_tokens = []  # python表达式的token
        self.completed = False  # 是否完成生成

class Engine:
    """
    高效推理引擎（主类）
    
    功能：
        - 高效的token生成（KV缓存、批量采样）
        - 流式生成（逐token输出）
        - 批量生成（返回完整序列）
        - 工具使用集成（计算器）
    
    关键优化：
        1. 一次预填充，多次采样：
           - 用batch=1预填充提示词一次
           - 复制KV缓存到batch=N并行生成多个样本
           - 避免重复计算提示词（大幅提速）
        
        2. KV缓存复用：
           - 缓存历史token的键值对
           - 每步只计算新token的KV
           - 减少计算量和内存访问
        
        3. 动态token注入：
           - 工具输出强制注入到生成序列
           - 避免模型生成错误的工具输出
    
    方法：
        generate: 流式生成（逐token返回）
        generate_batch: 批量生成（返回完整序列）
    """

    def __init__(self, model, tokenizer):
        """
        初始化推理引擎
        
        参数：
            model: GPT模型实例
            tokenizer: 分词器（用于工具使用：编码/解码表达式）
        """
        self.model = model
        self.tokenizer = tokenizer  # 工具使用需要（编码/解码计算器表达式）

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        流式生成token（逐token返回）
        
        参数：
            tokens: 提示词token列表（int列表）
            num_samples: 并行生成的样本数（默认1）
            max_tokens: 最大生成token数（None=无限制）
            temperature: 采样温度（默认1.0）
            top_k: Top-K采样（None=全词汇表）
            seed: 随机种子（用于可重复性）
        
        生成（yield）：
            token_column: 每行的下一个token（int列表，长度=num_samples）
            token_masks: 每行的mask（0=强制token，1=采样token）
        
        核心流程：
            1. 预填充阶段（batch=1）：
               - 前向传播提示词，获取最后一个token的logits
               - 采样第一个生成token
               - 缓存KV到kv_cache_prefill
            
            2. KV缓存复制阶段：
               - 创建batch=num_samples的kv_cache_decode
               - 从kv_cache_prefill预填充
               - 广播到所有样本
            
            3. 主生成循环：
               - 检查停止条件（max_tokens、所有行完成）
               - 前向传播当前token列
               - 采样下一个token（或使用强制token）
               - 更新每行状态（RowState）
               - 处理工具使用（python_start/end、output_start/end）
               - yield token_column和masks
        
        工具使用流程：
            - 遇到<|python_start|> → 开始累积表达式token
            - 遇到<|python_end|> → 计算表达式，强制注入结果
            - 结果格式：<|output_start|> + 结果token + <|output_end|>
        
        终止条件：
            - 遇到<|assistant_end|>或<|bos|>：标记行完成
            - max_tokens达到：停止生成
            - 所有行完成：退出循环
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "期望int列表"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)  # 设置随机种子（可重复性）

        # 获取工具使用状态机所需的特殊token
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")  # 计算器代码块开始
        python_end = get_special("<|python_end|>")  # 计算器代码块结束
        output_start = get_special("<|output_start|>")  # 计算器输出开始
        output_end = get_special("<|output_end|>")  # 计算器输出结束
        assistant_end = get_special("<|assistant_end|>")  # 助手回复结束（终止生成）
        bos = self.tokenizer.get_bos_token_id()  # 句子开始（终止生成）

        # ====================
        # 1) 预填充阶段（batch=1）
        # ====================
        # 用batch=1前向传播提示词，获取第一个生成token
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, T)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)  # (1, T, vocab_size)
        logits = logits[:, -1, :]  # 取最后一个token的logits (1, vocab_size)
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (1, 1)
        sampled_tokens = next_ids[:, 0].tolist()  # 转为列表

        # ====================
        # 2) KV缓存复制阶段
        # ====================
        # 创建batch=num_samples的KV缓存，并从预填充缓存复制数据
        # 这样避免了对每个样本重复预填充提示词
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)  # 复制预填充的KV缓存
        del kv_cache_prefill  # 释放预填充缓存（节省内存）

        # ====================
        # 3) 初始化每个样本的状态
        # ====================
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # ====================
        # 4) 主生成循环
        # ====================
        num_generated = 0
        first_iteration = True
        while True:
            # 停止条件1：达到最大token数
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # 停止条件2：所有行都已完成
            if all(state.completed for state in row_states):
                break

            # 获取采样的token（第一次迭代使用预填充的token，之后前向传播）
            if first_iteration:
                # 第一次迭代：使用预填充时已采样的token
                sampled_tokens = [sampled_tokens[0]] * num_samples  # 广播到所有行
                # TODO: 应该为每行采样一个token而不是广播
                first_iteration = False
            else:
                # 后续迭代：前向传播模型并为每行获取下一个token
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # 取最后一个时间步的logits (B, vocab_size)
                next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()  # 转为列表

            # 处理每一行：选择下一个token、更新状态、处理工具使用
            token_column = []  # 每行的下一个token ID
            token_masks = []  # 每行的mask（0=强制，1=采样）
            for i, state in enumerate(row_states):
                # 选择该行的下一个token
                is_forced = len(state.forced_tokens) > 0  # 队列中是否有待强制的token？
                token_masks.append(0 if is_forced else 1)  # mask: 0=强制，1=采样
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                
                # 更新该行的状态，添加新token
                state.current_tokens.append(next_token)
                
                # 遇到<|assistant_end|>或<|bos|>：标记行完成
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                
                # 处理工具逻辑（计算器状态机）
                if next_token == python_start:
                    # 进入python代码块
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    # 退出python代码块：计算表达式并强制注入结果
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            # 强制注入：<|output_start|> + 结果 + <|output_end|>
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    # 在python代码块内：累积表达式token
                    state.python_expr_tokens.append(next_token)

            # 生成token列（每行的下一个token）
            yield token_column, token_masks
            num_generated += 1
            
            # 准备下一次迭代的输入
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)  # (B, 1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        批量生成（非流式）
        
        参数：
            tokens: 提示词token列表
            num_samples: 并行生成的样本数
            **kwargs: 传递给generate()的其他参数（max_tokens、temperature、top_k、seed）
        
        返回：
            results: 生成的token序列列表（list of list of int）
            masks: 每个token的mask列表（0=强制，1=采样）
        
        特点：
            - 非流式：收集所有生成的token后一次性返回
            - 排除终止token：assistant_end和bos不包含在结果中
            - 返回masks：区分采样token和强制token（用于RL训练）
        
        用途：
            - 评估任务：需要完整序列
            - RL训练：需要完整序列和masks计算奖励
            - 非交互场景：不需要逐token显示
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        
        # 初始化结果列表（每个样本初始化为提示词）
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]  # 提示词的mask都是0（非采样）
        completed = [False] * num_samples
        
        # 迭代生成过程，收集所有token
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        # 遇到终止token：标记完成，但不添加到结果
                        completed[i] = True
                    else:
                        # 添加生成的token和mask
                        results[i].append(token)
                        masks[i].append(mask)
            # 所有行完成：提前退出
            if all(completed):
                break
        
        return results, masks


# =============================================================================
# 主程序：快速内联测试
# =============================================================================
if __name__ == "__main__":
    """
    快速内联测试：验证Engine与模型原生generate()的等价性
    
    目的：
        确保优化的Engine.generate()与朴素的model.generate()
        产生相同的token序列（在temperature=0.0时）
    
    测试流程：
        1. 加载base模型和分词器
        2. 用model.generate()生成参考序列（记录时间）
        3. 用Engine.generate()生成测试序列（记录时间）
        4. 比较两个序列是否完全一致
        5. 输出时间对比（Engine应该更快）
    
    预期结果：
        - 序列完全匹配（temperature=0.0保证确定性）
        - Engine速度更快（KV缓存优化）
    """
    import time
    
    # 初始化计算环境
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    
    # 加载模型和分词器
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    
    # 通用超参数（temperature=0.0保证确定性）
    kwargs = dict(max_tokens=64, temperature=0.0)
    
    # 设置测试提示词
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    
    # ====================
    # 参考实现：model.generate()
    # ====================
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"参考实现时间: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    
    # ====================
    # 优化实现：Engine.generate()
    # ====================
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs)  # 注意：在fp32下运行
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0]  # 只打印第一行
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine时间: {t1 - t0:.2f}s")
    
    # ====================
    # 比较两个序列
    # ====================
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"不匹配位置{i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"匹配结果: {reference_ids == generated_tokens}")
