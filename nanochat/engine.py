"""
高效推理引擎

本模块提供了一个优化的推理引擎，用于模型的高效文本生成。

核心设计理念：
    - 一切围绕token序列展开
    - 用户发送token序列给引擎
    - 引擎返回下一个token
    - 引擎不涉及分词，仅处理token ID序列

核心功能：
    1. KVCache: KV缓存管理（加速自回归生成）
    2. sample_next_token: 采样策略（温度、top-k）
    3. RowState: 每行生成状态跟踪
    4. Engine: 主推理引擎（支持批量生成、工具调用）

工具支持：
    - Calculator（计算器）：支持数学表达式和字符串操作
    - Python代码块：自动执行并注入结果

性能优化：
    - KV cache复用（减少重复计算）
    - 批量prefill（多样本并行生成）
    - 动态cache扩展（内存高效）
    - 流式生成（低延迟）
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model

# =============================================================================
# 计算器工具辅助函数
# =============================================================================

@contextmanager
def timeout(duration, formula):
    """
    超时上下文管理器
    
    参数：
        duration: 超时时长（秒）
        formula: 正在评估的公式（用于错误消息）
    
    功能：
        在duration秒后中断代码执行，抛出异常
    
    使用方式：
        with timeout(3, "1+2"):
            # 这里的代码最多执行3秒
    """
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': 执行超时（{duration}秒）")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    """
    带超时的安全表达式评估
    
    参数：
        formula: 要评估的Python表达式
        max_time: 最大执行时间（默认3秒）
    
    返回：
        - 评估结果（成功时）
        - None（失败或超时）
    
    安全特性：
        - 禁用所有内置函数（__builtins__={}）
        - 超时保护（防止无限循环）
        - 异常捕获（防止崩溃）
        - 忽略语法警告
    """
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                # 在空环境中评估（无内置函数，无变量）
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)  # 取消alarm
        # 静默失败（计算器错误使用是可以接受的）
        return None

def use_calculator(expr):
    """
    安全地评估Python表达式（计算器工具）
    
    参数：
        expr: 表达式字符串
    
    返回：
        - 评估结果（成功时）
        - None（失败或不安全）
    
    支持的操作：
        1. 数学表达式：
           - 支持：+ - * / ( ) 和数字
           - 不支持：** (幂运算被禁用)
           
        2. 字符串操作：
           - 支持：.count() 方法
           - 示例："hello".count("l") -> 2
    
    安全限制：
        - 白名单字符检查
        - 禁用危险关键字（import, exec, eval等）
        - 超时保护
        - 隔离执行环境
    """
    # 移除数字中的逗号（如1,000 -> 1000）
    expr = expr.replace(",", "")

    # 情况1：纯数学表达式
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # 禁用幂运算（防止指数爆炸）
            return None
        return eval_with_timeout(expr)

    # 情况2：字符串操作
    # 允许：字符串（单/双引号）、.count()、字母、数字、空格、括号
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # 禁用危险模式（安全检查）
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # 目前只允许.count()方法（未来可扩展）
    if '.count(' not in expr:
        return None

    # 带超时评估
    return eval_with_timeout(expr)

# =============================================================================
# KV缓存管理
# =============================================================================

class KVCache:
    """
    KV缓存管理器
    
    与GPT模型配合使用，维护注意力层的Key/Value缓存。
    
    作用：
        在自回归生成时，避免重复计算已生成token的K和V。
        通过缓存历史K/V，只需计算新token的K/V，大幅加速推理。
    
    核心属性：
        kv_cache: 缓存张量，形状(num_layers, 2, B, H, T, D)
            - num_layers: Transformer层数
            - 2: K和V（索引0=K, 1=V）
            - B: 批次大小
            - H: 注意力头数
            - T: 序列长度（动态增长）
            - D: 每个头的维度
        pos: 当前缓存位置（时间步）
    
    特性：
        - 懒初始化：首次插入时才分配内存
        - 动态扩展：空间不足时自动扩展（1024的倍数）
        - 批次扩展：支持从batch=1预填充到batch=N
        - 自动位置跟踪：最后一层插入后自动前进pos
    
    注意：
        pos在最后一层Transformer插入K/V后自动前进。
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        """
        初始化KV缓存
        
        参数：
            batch_size: 批次大小
            num_heads: 注意力头数
            seq_len: 初始序列长度（可动态增长）
            head_dim: 每个头的维度
            num_layers: Transformer层数
        """
        # K和V的形状（每层一个）
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None  # 懒初始化（首次插入时分配）
        self.pos = 0  # 当前缓存位置（时间步）

    def reset(self):
        """重置缓存位置到0（不释放内存）"""
        self.pos = 0

    def get_pos(self):
        """获取当前缓存位置"""
        return self.pos

    def prefill(self, other):
        """
        从另一个KV缓存预填充（支持批次扩展）
        
        参数：
            other: 源KV缓存对象
        
        用途：
            用于batch=1预填充后，从该点并行生成多个样本。
            例如：
                1. 用batch=1运行prompt的prefill
                2. 将结果复制到batch=N的cache中
                3. 从该点并行生成N个不同的样本
        
        批次扩展：
            如果self的batch_size > other的batch_size，会自动广播
            （other的batch_size必须是1）
        
        形状验证：
            - num_layers, k/v, num_heads, head_dim必须匹配
            - batch_size可扩展（other必须是1或相同）
            - seq_len: self >= other（必须有足够空间）
        """
        # 1) 验证形状
        assert self.kv_cache is None, "无法预填充非空的KV缓存"
        assert other.kv_cache is not None, "无法从None缓存预填充"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            # ix 0: num_layers, 1: k/v, 2: batch_size, 3: num_heads, 4: seq_len, 5: head_dim
            if ix in [0, 1, 3, 5]:
                # num_layers, k/v, num_heads, head_dim必须匹配
                assert dim1 == dim2, f"维度{ix}不匹配: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size可扩展
                assert dim1 == dim2 or dim2 == 1, f"批次维度不匹配: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len: self必须 >= other
                assert dim1 >= dim2, f"序列长度不匹配: {dim1} < {dim2}"
        
        # 2) 初始化缓存
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        
        # 3) 复制数据（自动广播batch维度）
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        
        # 4) 更新位置
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        """
        插入新的K/V到缓存，并返回完整的缓存视图
        
        参数：
            layer_idx: 当前Transformer层索引
            k: 新的Key张量，形状(B, H, T_add, D)
            v: 新的Value张量，形状(B, H, T_add, D)
        
        返回：
            (key_view, value_view): 完整的K/V缓存视图
                - key_view: 形状(B, H, t1, D)，包含所有历史K
                - value_view: 形状(B, H, t1, D)，包含所有历史V
                - t1 = self.pos + T_add
        
        特性：
            1. 懒初始化：首次调用时才分配内存
            2. 动态扩展：空间不足时自动增长（1024的倍数）
            3. 自动位置跟踪：最后一层后自动前进pos
        
        注意：
            只有在最后一层（layer_idx == num_layers - 1）插入后，
            pos才会前进，确保所有层使用相同的pos。
        """
        # 懒初始化缓存（需要知道dtype/device）
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        
        # 获取新K/V的形状
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        
        # 动态扩展缓存（如果空间不足）
        if t1 > self.kv_cache.size(4):
            # 计算需要的总长度：当前需求 + 1024缓冲
            t_needed = t1 + 1024
            # 向上取整到1024的倍数（对齐以提高性能）
            t_needed = (t_needed + 1023) & ~1023
            # 创建额外的缓存空间
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            # 拼接到现有缓存
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        
        # 插入新的K/V到缓存
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        
        # 返回完整的缓存视图（从0到t1）
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        
        # 在最后一层后前进位置
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        
        return key_view, value_view


# =============================================================================
# 采样策略
# =============================================================================

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """
    从logits采样下一个token
    
    参数：
        logits: 输出logits，形状(B, vocab_size)
        rng: 随机数生成器（torch.Generator）
        temperature: 采样温度（默认1.0）
            - 0.0: 贪婪采样（选择概率最大的token）
            - (0, 1): 更确定性（降低随机性）
            - 1.0: 标准采样（按原始概率分布）
            - >1.0: 更随机（增加随机性）
        top_k: Top-K采样（可选）
            - None: 禁用（从所有token中采样）
            - k: 只从概率最大的k个token中采样
    
    返回：
        next_token: 采样的token ID，形状(B, 1)
    
    采样策略：
        1. temperature=0: 贪婪（argmax）
        2. top_k=None: 全局采样（所有token）
        3. top_k=k: Top-K采样（只考虑top k个token）
    """
    assert temperature >= 0.0, "temperature必须非负"
    
    # 贪婪采样（temperature=0）
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # Top-K采样
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        # 获取top-k个最大的logits和对应的索引
        vals, idx = torch.topk(logits, k, dim=-1)
        # 温度缩放
        vals = vals / temperature
        # 计算概率分布
        probs = F.softmax(vals, dim=-1)
        # 从top-k中采样
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        # 映射回原始词汇表索引
        return idx.gather(1, choice)
    else:
        # 全局采样（所有token）
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# =============================================================================
# 生成状态跟踪
# =============================================================================

class RowState:
    """
    单行生成状态跟踪器
    
    在批量生成时，每一行（样本）都有独立的状态。
    这个类维护每行的生成状态，包括：
        - 已生成的token序列
        - 待强制注入的token队列（工具调用结果）
        - Python代码块状态（工具使用）
        - 完成标志
    
    属性：
        current_tokens: 当前token序列（list of int）
        forced_tokens: 强制注入的token队列（deque）
        in_python_block: 是否在Python代码块内（bool）
        python_expr_tokens: 当前Python表达式的token列表
        completed: 是否完成生成（bool）
    
    工具使用流程：
        1. 模型生成<|python_start|> -> in_python_block=True
        2. 累积Python表达式的token
        3. 模型生成<|python_end|> -> 执行表达式
        4. 将结果token注入forced_tokens队列
        5. 后续步骤强制使用这些token
    """
    def __init__(self, current_tokens=None):
        """
        初始化行状态
        
        参数：
            current_tokens: 初始token序列（默认为空列表）
        """
        self.current_tokens = current_tokens or []  # 当前token序列
        self.forced_tokens = deque()  # 待强制注入的token队列
        self.in_python_block = False  # 是否在Python块内
        self.python_expr_tokens = []  # 当前Python表达式的token
        self.completed = False  # 是否完成生成

class Engine:
    """
    高效推理引擎（主类）
    
    提供优化的文本生成功能，支持：
        - 批量并行生成（多样本）
        - KV缓存加速
        - 工具调用（计算器）
        - 流式输出
    
    属性：
        model: GPT模型实例
        tokenizer: 分词器实例（工具使用需要）
    
    核心方法：
        generate: 流式生成（yield每个token）
        generate_batch: 批量生成（返回完整序列）
    
    优化策略：
        1. 单次prefill + KV cache复制（多样本并行）
        2. 动态cache扩展（内存高效）
        3. 工具调用自动注入（无需模型重新生成）
    """

    def __init__(self, model, tokenizer):
        """
        初始化推理引擎
        
        参数：
            model: GPT模型实例
            tokenizer: 分词器实例（工具使用需要）
        """
        self.model = model
        self.tokenizer = tokenizer  # 工具调用需要（解码Python表达式）

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        流式生成（支持批量多样本）
        
        参数：
            tokens: 初始token列表（prompt）
            num_samples: 并行生成的样本数（默认1）
            max_tokens: 最大生成token数（None=无限制）
            temperature: 采样温度（默认1.0）
            top_k: Top-K采样（None=禁用）
            seed: 随机种子（默认42）
        
        生成：
            (token_column, token_masks): 元组
                - token_column: 每行的下一个token（list of int，长度=num_samples）
                - token_masks: 每行的mask（0=强制，1=采样）
        
        优化策略：
            1. 单次prefill（batch=1）：
               - 只对prompt运行一次前向传播
               - 得到首个采样token
            
            2. KV cache复制（broadcast到num_samples）：
               - 从batch=1的cache复制到batch=num_samples
               - 避免重复计算prompt的KV
            
            3. 并行decode（batch=num_samples）：
               - 后续步骤并行生成所有样本
               - 每个样本独立采样，互不影响
        
        工具调用流程：
            1. 检测<|python_start|> -> 开始累积表达式
            2. 累积表达式token直到<|python_end|>
            3. 解码并执行表达式（use_calculator）
            4. 将结果注入forced_tokens队列
            5. 后续步骤优先使用forced_tokens
        
        停止条件：
            1. 达到max_tokens
            2. 所有行都完成（遇到<|assistant_end|>或<|bos|>）
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "期望整数列表"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # 获取工具使用状态机需要的特殊token
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")  # Python代码块开始
        python_end = get_special("<|python_end|>")      # Python代码块结束
        output_start = get_special("<|output_start|>")  # 工具输出开始
        output_end = get_special("<|output_end|>")      # 工具输出结束
        assistant_end = get_special("<|assistant_end|>")  # 助手回复结束（停止生成）
        bos = self.tokenizer.get_bos_token_id()         # 序列开始token（停止生成）

        # ========== 阶段1：Prefill（batch=1） ==========
        # 对prompt运行一次前向传播，获取首个token
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]  # 只取最后一个位置的logits
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # ========== 阶段2：复制KV cache到所有样本 ==========
        # 从batch=1复制到batch=num_samples，避免重复计算prompt
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # 释放prefill cache的内存

        # ========== 阶段3：初始化每个样本的状态 ==========
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # ========== 阶段4：主生成循环 ==========
        num_generated = 0
        first_iteration = True
        while True:
            # 停止条件1：达到最大token数
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # 停止条件2：所有行都已完成
            if all(state.completed for state in row_states):
                break

            # 获取采样token（从prefill或forward传播）
            if first_iteration:
                # 使用prefill阶段已采样的token
                sampled_tokens = [sampled_tokens[0]] * num_samples  # 广播首个token到所有行
                # TODO: 应该为每行独立采样而不是广播
                first_iteration = False
            else:
                # 前向传播模型，获取每行的下一个token
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size) 只取最后时间步
                next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()

            # 处理每一行：选择下一个token、更新状态、可选的工具使用
            token_column = []  # 每行的下一个token ID
            token_masks = []   # 每行的mask（采样=1，强制=0）
            for i, state in enumerate(row_states):
                # 选择该行的下一个token（优先使用forced_tokens）
                is_forced = len(state.forced_tokens) > 0  # 是否有待强制的token？
                token_masks.append(0 if is_forced else 1)  # 0=强制，1=采样
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                
                # 更新该行的状态
                state.current_tokens.append(next_token)
                
                # 检测结束token（<|assistant_end|>或<|bos|>）
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                
                # ===== 工具使用逻辑 =====
                if next_token == python_start:
                    # 进入Python代码块
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    # 退出Python代码块，执行表达式
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        # 解码表达式
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        # 执行计算器
                        result = use_calculator(expr)
                        if result is not None:
                            # 将结果token注入forced_tokens队列
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    # 在Python代码块内，累积token
                    state.python_expr_tokens.append(next_token)

            # yield当前列的token
            yield token_column, token_masks
            num_generated += 1
            # 准备下一轮迭代的输入
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        批量生成（非流式，返回完整序列）
        
        参数：
            tokens: 初始token列表（prompt）
            num_samples: 并行生成的样本数（默认1）
            **kwargs: 传递给generate的其他参数
        
        返回：
            (results, masks): 元组
                - results: token序列列表（list of lists of ints）
                - masks: mask列表（list of lists of ints）
                  每个mask: 0=prompt或强制token，1=采样token
        
        特性：
            - 非流式：等待所有token生成完毕后返回
            - 自动排除终止token（assistant_end, bos）
            - 返回完整序列和mask（便于分析）
        
        用途：
            - 批量评估
            - 需要完整序列的场景
            - 不需要流式输出的场景
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        
        # 初始化结果和mask（包含prompt）
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]  # prompt的mask全为0
        completed = [False] * num_samples
        
        # 调用流式生成，收集所有token
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        # 遇到终止token，标记完成（不添加该token）
                        completed[i] = True
                    else:
                        # 添加token和mask
                        results[i].append(token)
                        masks[i].append(mask)
            # 所有行完成后提前退出
            if all(completed):
                break
        
        return results, masks


if __name__ == "__main__":
    """
    快速内联测试
    
    目的：验证Engine.generate和model.generate的等价性
    
    测试步骤：
        1. 使用model.generate（朴素/慢速）生成参考序列
        2. 使用Engine.generate（优化/快速）生成测试序列
        3. 比较两个序列是否完全相同
        4. 比较执行时间
    
    预期结果：
        - 序列完全匹配（确保正确性）
        - Engine更快（验证优化有效）
    """
    import time
    
    # 初始化计算环境
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    
    # 加载模型和分词器
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    
    # 公共超参数
    kwargs = dict(max_tokens=64, temperature=0.0)
    
    # 设置prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    
    # ===== 参考实现：model.generate =====
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
    
    # ===== 优化实现：Engine.generate =====
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs)  # 注意：运行在fp32
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
    print(f"Engine实现时间: {t1 - t0:.2f}s")
    
    # ===== 比较结果 =====
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"不匹配位置 {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"序列匹配: {reference_ids == generated_tokens}")