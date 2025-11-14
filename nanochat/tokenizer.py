"""
GPT-4风格的BPE分词器

本模块提供两种BPE（Byte Pair Encoding）分词器实现：

1. HuggingFace Tokenizer（HuggingFaceTokenizer类）：
   - 优点：可以同时训练和推理，功能完整
   - 缺点：API比较复杂，文档混乱
   - 用途：备用实现，兼容性测试

2. RustBPE + tiktoken组合（RustBPETokenizer类）：
   - RustBPE：用于训练（高性能Rust实现）
   - tiktoken：用于推理（OpenAI的高效tokenizer）
   - 优点：训练快，推理快，API简洁
   - 用途：主要实现，推荐使用

特殊Token：
    - <|bos|>: 文档开始token（每个文档开头）
    - <|user_start|>/<|user_end|>: 用户消息边界
    - <|assistant_start|>/<|assistant_end|>: 助手消息边界
    - <|python_start|>/<|python_end|>: Python工具调用边界
    - <|output_start|>/<|output_end|>: Python输出边界

分割模式（SPLIT_PATTERN）：
    与GPT-4略有不同：使用\p{N}{1,2}而非\p{N}{1,3}
    原因：对小词汇表更友好（避免在数字上"浪费"太多token）
    注意：未经充分验证，待优化（TODO）
"""

import os  # 文件操作
import copy  # 深拷贝（避免修改原对象）
from functools import lru_cache  # LRU缓存装饰器

# 特殊token定义
SPECIAL_TOKENS = [
    # 每个文档以BOS（Beginning of Sequence）token开始，用于分隔文档
    "<|bos|>",
    # 以下token仅在微调时使用，用于将对话渲染为token ID
    "<|user_start|>",      # 用户消息开始
    "<|user_end|>",        # 用户消息结束
    "<|assistant_start|>",  # 助手消息开始
    "<|assistant_end|>",    # 助手消息结束
    "<|python_start|>",     # 助手调用Python REPL工具
    "<|python_end|>",       # Python调用结束
    "<|output_start|>",     # Python REPL输出回助手
    "<|output_end|>",       # 输出结束
]

# 注意：此分割模式与GPT-4略有不同，使用\p{N}{1,2}而非\p{N}{1,3}
# 原因：我不想为较小的词汇表在数字上"浪费"太多token
# 尚未验证这是否是个好主意，TODO
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# 基于HuggingFace Tokenizer的通用GPT-4风格分词器
from tokenizers import Tokenizer as HFTokenizer  # HuggingFace Tokenizer核心
from tokenizers import pre_tokenizers, decoders, Regex  # 预处理器、解码器、正则
from tokenizers.models import BPE  # BPE模型
from tokenizers.trainers import BpeTrainer  # BPE训练器

class HuggingFaceTokenizer:
    """
    HuggingFace Tokenizer的轻量级封装
    
    特性：
        - 封装HuggingFace Tokenizer的复杂API
        - 提供统一的训练和推理接口
        - 支持特殊token处理
        - 支持批量编码
    
    方法：
        - from_pretrained: 从HuggingFace Hub加载
        - from_directory: 从本地目录加载
        - train_from_iterator: 从文本迭代器训练
        - encode: 编码文本为token IDs
        - decode: 解码token IDs为文本
        - save: 保存到本地
    
    用途：
        备用实现，主要用于兼容性测试和对比验证
    """

    def __init__(self, tokenizer):
        """
        初始化
        
        参数：
            tokenizer: HuggingFace Tokenizer对象
        """
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        """
        从HuggingFace Hub加载预训练分词器
        
        参数：
            hf_path: HuggingFace Hub路径（如"gpt2"）
        
        返回：
            HuggingFaceTokenizer实例
        """
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """
        从本地目录加载分词器
        
        参数：
            tokenizer_dir: 本地目录路径（如"out/tokenizer"）
        
        返回：
            HuggingFaceTokenizer实例
        """
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """
        从文本迭代器训练BPE分词器
        
        参数：
            text_iterator: 文本迭代器（生成字符串）
            vocab_size: 目标词汇表大小
        
        返回：
            训练好的HuggingFaceTokenizer实例
        
        配置：
            - BPE模型（byte_fallback=True）
            - 无Normalizer
            - GPT-4风格Pre-tokenizer（正则分割+ByteLevel）
            - ByteLevel解码器
            - 无Post-processor
        """
        # 配置HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,  # 必需！字节回退
            unk_token=None,
            fuse_unk=False,
        ))
        
        # Normalizer: None（不进行文本归一化）
        tokenizer.normalizer = None
        
        # Pre-tokenizer: GPT-4风格
        # GPT-4使用的正则模式，在BPE之前将文本分割成组
        # 注意：模式从\p{N}{1,3}改为\p{N}{1,2}，因为我怀疑对小模型和小词汇表有害
        # 在token空间上有点浪费（但尚未验证！TODO）
        gpt4_split_regex = Regex(SPLIT_PATTERN)  # HuggingFace要求包装在Regex中！
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        
        # Decoder: ByteLevel（与ByteLevel pre-tokenizer配对）
        tokenizer.decoder = decoders.ByteLevel()
        
        # Post-processor: None（不进行后处理）
        tokenizer.post_processor = None
        
        # Trainer: BPE训练器
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,  # 显示进度条
            min_frequency=0,  # 无最小频率要求
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 初始字母表（256个字节）
            special_tokens=SPECIAL_TOKENS,  # 特殊token列表
        )
        
        # 启动训练
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        """获取词汇表大小"""
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        """获取所有特殊token列表"""
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        """将token ID转换为token字符串"""
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        """
        编码单个字符串（内部方法）
        
        参数：
            text: 输入文本（字符串）
            prepend: 前置token（特殊token字符串或token ID）
            append: 后置token（特殊token字符串或token ID）
        
        返回：
            ids: token ID列表
        """
        assert isinstance(text, str), "text必须是字符串"
        ids = []
        
        # 添加前置token
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        
        # 编码主文本（不自动添加特殊token）
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        
        # 添加后置token
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        
        return ids

    def encode_special(self, text):
        """通过精确匹配编码单个特殊token"""
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        """获取BOS token的ID"""
        bos = self.encode_special("<|bos|>")
        return bos

    def encode(self, text, *args, **kwargs):
        """
        编码文本为token IDs
        
        参数：
            text: 字符串或字符串列表
            *args, **kwargs: 传递给_encode_one的参数
        
        返回：
            - 如果text是字符串：返回token ID列表
            - 如果text是列表：返回token ID列表的列表
        """
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"无效的输入类型: {type(text)}")

    def __call__(self, *args, **kwargs):
        """使对象可调用，等同于encode()"""
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        解码token IDs为文本
        
        参数：
            ids: token ID列表
        
        返回：
            解码后的字符串（保留特殊token）
        """
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        """
        保存分词器到磁盘
        
        参数：
            tokenizer_dir: 目标目录路径
        
        输出：
            tokenizer.json文件
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"已保存分词器到 {tokenizer_path}")

# -----------------------------------------------------------------------------
# 基于rustbpe + tiktoken组合的分词器
import pickle  # 序列化（保存tiktoken编码器）
import rustbpe  # Rust BPE训练器（高性能）
import tiktoken  # OpenAI的高效tokenizer（推理）

class RustBPETokenizer:
    """
    RustBPE + tiktoken组合分词器（推荐实现）
    
    架构：
        - 训练：使用rustbpe（Rust实现，速度快）
        - 推理：使用tiktoken（OpenAI实现，效率高）
    
    优势：
        1. 训练速度：Rust实现比Python快10-100倍
        2. 推理效率：tiktoken经过高度优化
        3. API简洁：接口清晰，易于使用
        4. 兼容性：可导出为tiktoken格式
    
    方法：
        - train_from_iterator: 从文本迭代器训练
        - from_directory: 从本地加载
        - from_pretrained: 从tiktoken预训练模型加载
        - encode: 编码（支持批量）
        - decode: 解码
        - render_conversation: 渲染对话（SFT）
        - render_for_completion: 渲染对话前缀（RL）
    
    特殊功能：
        - 对话渲染：自动添加特殊token和mask
        - 工具使用：支持Python工具调用标记
        - 批量编码：多线程并行
    """

    def __init__(self, enc, bos_token):
        """
        初始化
        
        参数：
            enc: tiktoken Encoding对象
            bos_token: BOS token字符串（如"<|bos|>"）
        """
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """
        从文本迭代器训练BPE分词器
        
        参数：
            text_iterator: 文本迭代器（生成字符串）
            vocab_size: 目标词汇表大小（包括特殊token）
        
        返回：
            训练好的RustBPETokenizer实例
        
        训练流程：
            1. 使用rustbpe训练（Rust实现，速度快）
            2. 构造tiktoken Encoding（用于高效推理）
            3. 添加特殊token
        
        特殊token处理：
            - vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
            - 先训练256个字节 + BPE merges
            - 然后添加特殊token到词汇表末尾
        """
        # 1) 使用rustbpe训练
        tokenizer = rustbpe.Tokenizer()
        # 特殊token稍后在__init__中插入，这里不训练它们
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special必须至少256，得到{vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        
        # 2) 构造相关的tiktoken编码（用于推理）
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,  # dict[bytes, int] (token字节 -> 合并优先级排名)
            special_tokens=special_tokens,    # dict[str, int] (特殊token名称 -> token ID)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """
        从本地目录加载分词器
        
        参数：
            tokenizer_dir: 本地目录路径
        
        返回：
            RustBPETokenizer实例
        
        加载文件：
            tokenizer.pkl: pickle序列化的tiktoken Encoding对象
        """
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        """
        从tiktoken预训练模型加载
        
        参数：
            tiktoken_name: tiktoken模型名（如"gpt2"、"cl100k_base"）
        
        返回：
            RustBPETokenizer实例
        
        特殊token说明：
            tiktoken使用"<|endoftext|>"作为文档分隔符token
            这很令人困惑，因为此token几乎总是PREPENDED到文档开头
            它最常用于在推理时向LLM发信号表示新序列的开始
            所以在nanoChat中我们总是使用"<|bos|>"（"beginning of sequence"的缩写）
            但历史上它通常被称为"<|endoftext|>"
        
        参考：
            https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        """
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        """获取词汇表大小"""
        return self.enc.n_vocab

    def get_special_tokens(self):
        """获取所有特殊token集合"""
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        """将token ID解码为token字符串"""
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """
        编码单个特殊token（带LRU缓存）
        
        参数：
            text: 特殊token字符串
        
        返回：
            token ID
        
        缓存：
            使用LRU缓存避免重复查找（特殊token经常使用）
        """
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        """获取BOS token的ID"""
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """
        编码文本为token IDs
        
        参数：
            text: 字符串或字符串列表
            prepend: 前置token（特殊token字符串或token ID）
            append: 后置token（特殊token字符串或token ID）
            num_threads: 批量编码的线程数（默认8）
        
        返回：
            - 如果text是字符串：返回token ID列表
            - 如果text是列表：返回token ID列表的列表
        
        特性：
            - 单个字符串：使用encode_ordinary
            - 批量字符串：使用encode_ordinary_batch（多线程并行）
        
        TODO：
            prepend/append时的insert/append略低效，可以优化
        """
        # 预处理prepend/append
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        # 编码主文本
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)  # TODO: 这里略低效? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)  # TODO: 同上
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"无效的输入类型: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        """使对象可调用，等同于encode()"""
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        解码token IDs为文本
        
        参数：
            ids: token ID列表
        
        返回：
            解码后的字符串
        """
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        """
        保存分词器到磁盘
        
        参数：
            tokenizer_dir: 目标目录路径
        
        输出：
            tokenizer.pkl文件（pickle序列化的tiktoken Encoding对象）
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"已保存分词器编码到 {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        渲染单个对话为token IDs（用于SFT训练）
        
        参数：
            conversation: 对话字典，包含messages列表
            max_tokens: 最大token数（截断）
        
        返回：
            (ids, mask): token IDs列表和mask列表
            - ids: list[int] - token IDs
            - mask: list[int] - mask（1=训练，0=忽略）
        
        对话格式：
            {
                "messages": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    ...
                ]
            }
        
        特殊处理：
            1. 系统消息：合并到第一个用户消息
            2. 消息交替：user/assistant严格交替
            3. mask策略：
               - 用户消息：mask=0（不训练）
               - 助手消息：mask=1（训练）
               - Python输出：mask=0（来自工具）
            4. 特殊token：自动添加<|bos|>、角色标记等
        
        工具使用支持：
            Assistant内容可以是：
            - 字符串：简单文本
            - 列表：包含多个部分
              - {"type": "text", "text": "..."}
              - {"type": "python", "text": "..."}
              - {"type": "python_output", "text": "..."}
        """
        # 我们将返回的ids和masks，以及一个辅助函数来构建它们
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            """辅助函数：添加token到ids和mask列表"""
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # 有时第一条消息是系统消息...
        # => 只需将其与第二条（用户）消息合并
        if conversation["messages"][0]["role"] == "system":
            # 这里需要进行一些对话处理...
            conversation = copy.deepcopy(conversation)  # 避免修改原对象
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "系统消息后必须跟用户消息"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"对话消息少于1条: {messages}"

        # 获取我们需要的所有特殊token
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # 现在可以tokenize对话了
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # 这里进行一些合理性检查以防止错误
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"消息{i}来自{message['role']}，但应该来自{must_be_from}"

            # content可以是简单字符串或部分列表（例如包含工具调用）
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "用户消息应该是简单字符串"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # 简单字符串 => 直接添加token
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # 文本部分 => 直接添加token
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # Python工具调用 => 在<|python_start|>和<|python_end|>内添加token
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # Python输出 => 在<|output_start|>和<|output_end|>内添加token
                            # 这些token不被监督，因为token在测试时来自Python
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"未知的部分类型: {part['type']}")
                else:
                    raise ValueError(f"未知的内容类型: {type(content)}")
                add_tokens(assistant_end, 1)

        # 截断到max_tokens（帮助防止OOM）
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """
        可视化tokenization（调试辅助工具）
        
        参数：
            ids: token ID列表
            mask: mask列表
            with_token_id: 是否显示token ID（默认False）
        
        返回：
            彩色字符串（带ANSI颜色码）
        
        颜色方案：
            - 绿色：mask=1（训练token）
            - 红色：mask=0（忽略token）
            - 灰色：token ID（如果with_token_id=True）
        
        用途：
            调试render_conversation的输出，可视化哪些token被训练
        """
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        渲染对话前缀（用于强化学习）
        
        参数：
            conversation: 对话字典（包含messages列表）
        
        返回：
            ids: token ID列表（不返回mask）
        
        用途：
            在强化学习中，我们想渲染对话来为Assistant准备一个补全
            与Chat SFT情况不同，我们不需要返回mask
        
        处理步骤：
            1. 深拷贝对话（避免修改原对象）
            2. 移除最后一条消息（Assistant的）
            3. tokenize对话
            4. 追加<|assistant_start|> token
        
        最终格式：
            [对话历史] <|assistant_start|> [待生成]
            模型从这里开始生成Assistant的回复
        """
        # 需要做一些处理：移除最后一条消息（Assistant的）
        conversation = copy.deepcopy(conversation)  # 避免修改原对象
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "最后一条消息必须来自Assistant"
        messages.pop()  # 原地移除最后一条消息（Assistant的）

        # 现在tokenize对话
        ids, mask = self.render_conversation(conversation)

        # 最后，为了为Assistant准备补全，追加Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    """
    获取nanochat分词器（便捷函数）
    
    返回：
        RustBPETokenizer实例（从标准目录加载）
    
    标准目录：
        {NANOCHAT_BASE_DIR}/tokenizer/
    
    用途：
        在训练和推理脚本中快速获取分词器
    """
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)  # 备用
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    """
    获取token字节长度表（用于BPB计算）
    
    参数：
        device: 目标设备（"cpu"或"cuda"）
    
    返回：
        token_bytes: 1D张量，形状(vocab_size,)
        - token_bytes[i] = token i的UTF-8字节长度
        - 特殊token的字节长度为0（不计入BPB）
    
    文件来源：
        由tok_train.py生成，保存在{NANOCHAT_BASE_DIR}/tokenizer/token_bytes.pt
    
    用途：
        在evaluate_bpb中使用，计算与词汇表大小无关的损失指标
    """
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes未找到：{token_bytes_path}？应由tok_train.py生成"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
