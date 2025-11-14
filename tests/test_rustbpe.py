"""
BPE分词器实现对比测试

这个测试文件对比了多种BPE（Byte Pair Encoding）分词器的训练实现，
确保它们产生相同的词汇表、合并规则和编码结果。

测试对象：
    1. Python参考实现（非常慢，用于验证正确性）
    2. 优化的Python实现（使用原地修改和增量更新）
    3. HuggingFace tokenizers库的训练实现
    4. 我们的自定义RustBPE实现（最快）

测试目的：
    - 验证所有实现计算出相同的合并序列
    - 验证产生相同的词汇表
    - 验证对相同文本产生相同的tokenization结果
    - 验证可以导出到tiktoken并保持一致性

运行方式：
    python -m pytest tests/test_rustbpe.py -v -s
    
参数说明：
    -v: 详细模式（verbose），显示每个测试的详细信息
    -s: 显示print输出（show prints）

技术背景：
    BPE是一种数据压缩技术，广泛应用于NLP领域的分词。
    它通过迭代地合并最频繁出现的字节对来构建词汇表。
    我们使用tiktoken进行推理，因为它在Python中提供了最高效的编码性能。
"""

# regex库：支持Unicode属性的高级正则表达式
import regex as re
# Counter：计数器，defaultdict：默认字典
from collections import Counter, defaultdict
# time：时间测量
import time
# rustbpe：我们的Rust实现的BPE分词器
import rustbpe
# tiktoken：OpenAI的快速BPE实现
import tiktoken
# pytest：Python测试框架
import pytest

# GPT-4的分词模式：处理缩写、Unicode字符、数字、标点和空白
# 这个正则表达式定义了如何将文本切分成预分词块
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# =============================================================================
# Python参考实现分词器
# =============================================================================
# 这个实现主要来自minbpe项目，经过了一些精简
# 虽然速度很慢，但逻辑清晰，用于验证其他实现的正确性

def get_stats(ids, counts=None):
    """
    统计连续字节对的出现次数
    
    参数：
        ids: 整数列表（表示字节序列）
        counts: 可选的现有计数字典（用于累加）
    
    返回：
        字典，键为(int, int)的字节对，值为出现次数
    
    示例：
        [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # 迭代相邻元素
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    在整数列表中将所有连续出现的字节对替换为新token
    
    参数：
        ids: 整数列表
        pair: 要合并的字节对 (a, b)
        idx: 新token的ID
    
    返回：
        合并后的新整数列表
    
    示例：
        ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # 如果不在最后一个位置，且当前对匹配，则替换
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class RegexTokenizer:
    """
    基于正则表达式的BPE分词器（Python参考实现）
    
    这是一个完整但较慢的实现，用于验证其他优化实现的正确性。
    使用正则表达式将文本预分割成块，然后在每个块内进行BPE训练。
    """

    def __init__(self, pattern=None):
        """
        初始化分词器
        
        参数：
            pattern: 可选，预分割的正则表达式模式（默认使用GPT-4模式）
        
        属性：
            merges: 合并规则字典 {(int, int): int}
            special_tokens: 特殊token字典 {str: int}
                例如: {'<|endoftext|>': 100257}
            vocab: 词汇表 {int: bytes}
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.merges = {}  # (int, int) -> int
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """
        根据合并规则构建词汇表
        
        词汇表是从合并规则确定性地推导出来的：
        - 前256个ID对应单个字节
        - 后续ID通过合并规则递归构建
        - 特殊token附加到词汇表末尾
        
        返回：
            词汇表字典 {token_id: bytes}
        """
        # 基础词汇：256个单字节
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # 根据合并规则添加多字节token
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # 添加特殊token
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        """
        训练BPE分词器
        
        参数：
            text: 训练文本
            vocab_size: 目标词汇表大小（必须 >= 256）
            verbose: 是否打印详细训练信息
        
        返回：
            ambiguous: 布尔值，指示是否存在歧义合并（即多个字节对有相同的最大计数）
        
        算法流程：
            1. 将文本按正则表达式切分成块
            2. 将每个块编码为字节序列
            3. 迭代num_merges次：
                - 统计所有连续字节对的出现次数
                - 选择出现次数最多的字节对
                - 将该字节对合并为新token
                - 更新词汇表
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 跟踪训练过程中是否出现歧义合并（多个字节对有相同的最大计数）
        ambiguous = False

        # 使用正则表达式将文本切分成块
        text_chunks = re.findall(self.compiled_pattern, text)

        # 输入文本预处理：将每个文本块转换为字节列表
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # 迭代合并最常见的字节对以创建新token
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes
        
        for i in range(num_merges):
            # 统计所有连续字节对的出现次数
            stats = {}
            for chunk_ids in ids:
                # 传入stats会原地更新，累加计数
                get_stats(chunk_ids, stats)
            
            # 找到出现次数最多的字节对
            pair = max(stats, key=stats.get)
            
            # 检查合并是否存在歧义（即最大值不唯一）
            pair_count = stats[pair]
            pairs_with_max_count = [pair for pair, count in stats.items() if count == pair_count]
            if len(pairs_with_max_count) > 1:
                # 存在多个相同计数的字节对，合并顺序可能不确定
                ambiguous = True
            
            # 创建新token：分配下一个可用ID
            idx = 256 + i
            
            # 在所有出现位置替换该字节对为新token
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            
            # 保存合并规则
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            # 详细输出
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # 保存类变量
        self.merges = merges  # 用于encode()
        self.vocab = vocab    # 用于decode()
        return ambiguous

    def _encode_chunk(self, text_bytes):
        """
        对单个文本块进行编码
        
        参数：
            text_bytes: 字节序列
        
        返回：
            token ID列表
        
        算法：
            1. 将字节转换为整数列表（0-255）
            2. 循环查找可合并的字节对（按合并顺序）
            3. 应用合并规则直到无法继续合并
        """
        # 首先将所有字节转换为0-255范围的整数
        ids = list(text_bytes)
        while len(ids) >= 2:
            # 找到具有最低合并索引的字节对（即最早的合并规则）
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # 巧妙之处：如果没有更多的合并规则可用，key函数会对每个字节对返回inf，
            # min会任意返回列表中的第一个字节对
            # 我们可以通过成员检查来检测这种终止情况
            if pair not in self.merges:
                break  # 没有更多可以合并的了
            # 否则合并最佳字节对（最低合并索引）
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """
        编码文本（忽略特殊token）
        
        参数：
            text: 输入文本字符串
        
        返回：
            token ID列表
        
        过程：
            1. 使用正则表达式将文本分割成块
            2. 分别编码每个块
            3. 连接所有块的结果
        """
        # 按正则表达式模式将文本切分成块
        text_chunks = re.findall(self.compiled_pattern, text)
        # 分别编码所有文本块，然后连接结果
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # 原始字节
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

# =============================================================================
# 优化的Python分词器
# =============================================================================
# 这是参考实现的优化版本，使用原地修改和增量更新来提高性能

def fast_merge_inplace(ids, pair, idx):
    """
    原地合并：在整数列表中原地替换所有连续出现的字节对
    
    参数：
        ids: 整数列表（会被原地修改）
        pair: 要合并的字节对 (a, b)
        idx: 新token的ID
    
    返回：
        修改后的ids（为了链式调用）
    
    示例：
        ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    
    优化点：
        使用.pop()原地删除，避免创建新列表，减少内存分配
    """
    # 找到所有出现该字节对的位置
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i+1] == pair[1]:
            ids[i] = idx
            ids.pop(i+1)  # 原地删除
        else:
            i += 1
    return ids


class FastRegexTokenizer:
    """
    优化的基于正则表达式的BPE分词器
    
    相比参考实现，引入了多项优化：
    - 内联函数以减少函数调用开销
    - 使用.pop()原地修改列表而非创建新列表
    - 合并相同的文本块为唯一块
    - 增量更新计数（仅更新受影响的块）
    - 位置追踪以加速合并操作
    """

    def __init__(self, pattern=None):
        """
        初始化分词器
        
        参数：
            pattern: 可选，预分割的正则表达式模式（默认使用GPT-4模式）
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.merges = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """根据合并规则确定性地构建词汇表"""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        """
        训练BPE分词器（优化版本）
        
        引入的优化：
        - 通过内联函数减少函数调用开销
        - 使用.pop()原地修改ID列表而非创建新列表
        - 将相同的文本块合并为唯一块（大幅减少处理量）
        - 更智能地更新计数 - 只更新受影响的块周围
        - 使用位置追踪集合快速定位包含特定字节对的块
        
        参数：
            text: 训练文本
            vocab_size: 目标词汇表大小（必须 >= 256）
            verbose: 是否打印详细训练信息
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 使用正则表达式将文本切分成块
        text_chunks = re.findall(self.compiled_pattern, text)

        # 许多文本块是相同的，我们可以将它们"折叠"为唯一块
        # 这是一个重要的优化，可以大幅减少需要处理的数据量
        counts = Counter(text_chunks)
        unique_chunks = [ch for ch, count in counts.items()]
        chunk_counts = [count for ch, count in counts.items()]

        # 输入文本预处理
        ids = [list(ch.encode("utf-8")) for ch in unique_chunks]
        # 迭代合并最常见的字节对以创建新token
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes

        # 初始计数：构建统计信息和位置追踪
        stats = defaultdict(int)
        positions = defaultdict(set)  # pair -> 包含此字节对的chunk索引集合

        # 遍历所有唯一块，初始化统计和位置信息
        for chunk_idx, (chunk_ids, count) in enumerate(zip(ids, chunk_counts)):
            for pair in zip(chunk_ids, chunk_ids[1:]):
                stats[pair] += count  # 加权计数（乘以块的重复次数）
                positions[pair].add(chunk_idx)  # 记录包含此字节对的块

        # ========== 主合并循环 ==========
        for i in range(num_merges):
            if not stats:
                break

            # 找到出现次数最多的字节对
            pair = max(stats, key=stats.get)
            # 创建新token：分配下一个可用ID
            idx = 256 + i

            # 获取包含此字节对的所有块（关键优化：只处理受影响的块）
            affected_chunks = positions[pair]

            # 追踪计数变化以进行增量更新（核心优化）
            count_changes = defaultdict(int)

            # 只在受影响的块中替换字节对的所有出现
            for chunk_idx in affected_chunks:
                chunk_ids = ids[chunk_idx]
                chunk_count = chunk_counts[chunk_idx]  # 此块的重复次数
                ix = 0
                while ix < len(chunk_ids) - 1:
                    if chunk_ids[ix] == pair[0] and chunk_ids[ix+1] == pair[1]:
                        # 追踪正在被移除/添加的字节对
                        # 移除: (prev, A), (A, B), (B, next)
                        # 其中(A, B)是要合并的字节对
                        
                        # 如果不在开头，移除左侧字节对 (prev, A)
                        if ix > 0:
                            old_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[old_left] -= chunk_count

                        # 被合并的字节对消失
                        count_changes[pair] -= chunk_count

                        # 如果不在倒数第二位，移除右侧字节对 (B, next)
                        if ix + 2 < len(chunk_ids):
                            old_right = (chunk_ids[ix+1], chunk_ids[ix+2])
                            count_changes[old_right] -= chunk_count

                        # 应用合并：将(A, B)替换为C
                        chunk_ids[ix] = idx
                        chunk_ids.pop(ix+1)  # 原地删除

                        # 添加: (prev, C), (C, next)
                        # 新的字节对出现
                        
                        # 如果不在开头，添加新的左侧字节对 (prev, C)
                        if ix > 0:
                            new_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[new_left] += chunk_count

                        # 如果不在末尾，添加新的右侧字节对 (C, next)
                        if ix + 1 < len(chunk_ids):
                            new_right = (chunk_ids[ix], chunk_ids[ix+1])
                            count_changes[new_right] += chunk_count
                    else:
                        ix += 1

            # 应用增量变化到统计信息和位置（关键优化：避免重新计算所有块）
            for changed_pair, delta in count_changes.items():
                if changed_pair == pair:
                    # 被合并的字节对应该完全消失
                    continue

                stats[changed_pair] += delta

                # 更新变化字节对的位置 - 只检查受影响的块
                for chunk_idx in affected_chunks:
                    chunk_ids = ids[chunk_idx]
                    contains_pair = any((chunk_ids[j], chunk_ids[j+1]) == changed_pair
                                      for j in range(len(chunk_ids) - 1))
                    if contains_pair:
                        positions[changed_pair].add(chunk_idx)
                    else:
                        positions[changed_pair].discard(chunk_idx)

            # 完全移除被合并的字节对
            del stats[pair]
            del positions[pair]

            # 保存合并规则
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # 保存类变量
        self.merges = merges  # 用于encode()
        self.vocab = vocab    # 用于decode()

    def register_special_tokens(self, special_tokens):
        """
        注册特殊token
        
        参数：
            special_tokens: 字典 {str: int}
                例如: {"<|endoftext|>": 100257}
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        """
        解码token ID序列为文本
        
        参数：
            ids: token ID列表
        
        返回：
            解码后的文本字符串
        """
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        """对单个文本块进行编码（使用优化的原地合并）"""
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = fast_merge_inplace(ids, pair, idx)  # 使用优化的原地合并
        return ids

    def encode_ordinary(self, text):
        """编码文本（忽略特殊token）"""
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

# =============================================================================
# HuggingFace tokenizers库封装
# =============================================================================
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """
    HuggingFace Tokenizer的轻量级封装
    
    用于对比测试，验证我们的实现与HuggingFace的实现结果一致。
    HuggingFace tokenizers是用Rust编写的，性能很好，但字节顺序可能不同。
    """

    def __init__(self, tokenizer):
        """
        初始化封装器
        
        参数：
            tokenizer: HuggingFace的Tokenizer实例
        """
        self.tokenizer = tokenizer

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """
        从文本迭代器训练分词器
        
        参数：
            text_iterator: 文本迭代器
            vocab_size: 目标词汇表大小
        
        返回：
            HuggingFaceTokenizer实例
        
        配置说明：
            - BPE模型with byte_fallback（必需）
            - 无Normalizer
            - GPT-4风格的Pre-tokenizer
            - ByteLevel解码器
            - 无Post-processor
        """
        # 配置HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,  # 必需：支持字节级回退
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer：无（不做文本标准化）
        tokenizer.normalizer = None
        # Pre-tokenizer：GPT-4风格
        gpt4_split_regex = Regex(GPT4_SPLIT_PATTERN)  # HuggingFace要求包装在Regex中
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder：ByteLevel（与ByteLevel pre-tokenizer配对）
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor：无
        tokenizer.post_processor = None
        # Trainer：BPE训练器
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,  # 无最小频率要求
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],  # 无特殊token
        )
        # 启动训练
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def encode_ordinary(self, text):
        """编码文本（不添加特殊token）"""
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return ids

# =============================================================================
# 测试函数
# =============================================================================

@pytest.fixture(scope="module")
def enwik8_path():
    """
    pytest fixture：下载并缓存enwik8数据集
    
    enwik8是一个100MB的Wikipedia文本数据集，常用于压缩和分词基准测试。
    """
    import os
    import zipfile
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    # 下载并解压enwik8到.cache目录
    enwik8_url = "https://mattmahoney.net/dc/enwik8.zip"
    enwik8_local_path = os.path.join(base_dir, "enwik8")
    enwik8_local_path_zip = os.path.join(base_dir, "enwik8.zip")
    if not os.path.exists(enwik8_local_path):
        print(f"Downloading enwik8 to {enwik8_local_path_zip}")
        import requests
        response = requests.get(enwik8_url)
        with open(enwik8_local_path_zip, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(enwik8_local_path_zip, "r") as zip_ref:
            zip_ref.extractall(base_dir)
        print(f"Unzipped enwik8 to {enwik8_local_path}")
        os.remove(enwik8_local_path_zip)
        print(f"Removed {enwik8_local_path_zip}")
    else:
        print(f"Using existing enwik8 at {enwik8_local_path}")
    return enwik8_local_path


@pytest.fixture(scope="module")
def enwik8_small(enwik8_path):
    """pytest fixture：提供100KB的enwik8数据用于快速测试"""
    with open(enwik8_path, "r", encoding="utf-8") as f:
        return f.read(100_000)

@pytest.fixture(scope="module")
def enwik8_large(enwik8_path):
    """pytest fixture：提供10MB的enwik8数据用于性能测试"""
    with open(enwik8_path, "r", encoding="utf-8") as f:
        return f.read(10**7)

def time_function(func, *args, **kwargs):
    """
    测量函数执行时间
    
    参数：
        func: 要测量的函数
        *args, **kwargs: 传递给函数的参数
    
    返回：
        (result, elapsed): 函数结果和运行时间（秒）
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    return result, elapsed

def test_correctness(enwik8_small):
    """
    正确性测试：验证所有分词器实现产生相同的结果
    
    测试流程：
        1. 训练慢速参考实现（Python）
        2. 训练快速参考实现（优化的Python）
        3. 训练HuggingFace实现
        4. 训练RustBPE实现
        5. 验证所有实现产生相同的编码结果
        6. 验证RustBPE可以导出到tiktoken并保持一致
    
    参数：
        enwik8_small: 100KB的enwik8测试数据（来自fixture）
    """
    text = enwik8_small
    encode_text = text
    vocab_size = 256 + 20  # 基础256字节 + 20次合并

    # ========== 训练慢速参考实现 ==========
    print("\nTraining slow reference...")
    slow_reference_tokenizer = RegexTokenizer()
    ambiguous_flag, slow_reference_train_time = time_function(slow_reference_tokenizer.train, text, vocab_size)
    slow_reference_ids, slow_reference_encode_time = time_function(slow_reference_tokenizer.encode_ordinary, encode_text)
    print(f"Slow reference train time: {slow_reference_train_time:.4f}s")
    print(f"Slow reference encode time: {slow_reference_encode_time:.4f}s")
    print(slow_reference_ids[:20])

    # 检查是否存在歧义合并
    if ambiguous_flag:
        print("‼️ WARNING: merge order was detected to be ambiguous given current text and vocab size")
        print("The implementation could be correct but we might see different results below")
    else:
        print("✅ Merge order is NOT ambiguous")

    # ========== 训练快速参考实现 ==========
    print("\nTraining fast reference...")
    fast_reference_tokenizer = FastRegexTokenizer()
    _, fast_reference_train_time = time_function(fast_reference_tokenizer.train, text, vocab_size)
    fast_reference_ids, fast_reference_encode_time = time_function(fast_reference_tokenizer.encode_ordinary, encode_text)
    print(f"Fast reference train time: {fast_reference_train_time:.4f}s")
    print(f"Fast reference encode time: {fast_reference_encode_time:.4f}s")
    print(fast_reference_ids[:20])

    # 验证快速版本与慢速版本一致
    assert fast_reference_ids == slow_reference_ids, "Fast reference should match slow reference"
    print("✅ Fast == Slow")

    # ========== 训练HuggingFace实现 ==========
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(HuggingFaceTokenizer.train_from_iterator, [text], vocab_size)
    hf_ids, hf_encode_time = time_function(hf_tokenizer.encode_ordinary, encode_text)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    print(f"HuggingFace encode time: {hf_encode_time:.4f}s")
    print(hf_ids[:20])

    # HuggingFace使用不同的字节顺序，所以需要自定义匹配逻辑
    def custom_match(ids1, ids2):
        """
        自定义匹配函数：考虑HuggingFace的字节顺序可能不同
        
        规则：
        - 单字节token（<256）可以有不同的映射（字节顺序不同）
        - 合并token（>=256）必须完全相同
        """
        perm = {}
        for x, y in zip(ids1, ids2):
            if x < 256:
                if x in perm:
                    if perm[x] != y:
                        return False
                perm[x] = y
            if x >= 256 and x != y:
                return False
        return True

    assert custom_match(hf_ids, fast_reference_ids), "HuggingFace should match fast reference"
    print("✅ HuggingFace == Fast")

    # ========== 训练我们的Rust实现 ==========
    print("\nTraining rustbpe...")
    rustbpe_tokenizer = rustbpe.Tokenizer()
    _, rustbpe_train_time = time_function(rustbpe_tokenizer.train_from_iterator, [text], vocab_size)
    rustbpe_ids, rustbpe_encode_time = time_function(rustbpe_tokenizer.encode, encode_text)
    print(f"RustBPE train time: {rustbpe_train_time:.4f}s")
    print(f"RustBPE encode time: {rustbpe_encode_time:.4f}s")
    print(rustbpe_ids[:20])

    assert rustbpe_ids == fast_reference_ids, "RustBPE should match fast reference"
    print("✅ RustBPE == Fast")

    # ========== 测试导出到tiktoken ==========
    # 对于生产环境推理，我们使用tiktoken以获得最佳性能
    print("\nTesting tiktoken export...")
    pattern = rustbpe_tokenizer.get_pattern()
    mergeable_ranks_list = rustbpe_tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )
    tiktoken_ids, tiktoken_encode_time = time_function(enc.encode, encode_text)
    print(f"Tiktoken encode time: {tiktoken_encode_time:.4f}s")
    print(tiktoken_ids[:20])

    assert tiktoken_ids == rustbpe_ids, "Tiktoken should match RustBPE"
    print("✅ Tiktoken == RustBPE")


@pytest.mark.slow
def test_training_performance(enwik8_large):
    """
    性能测试：使用大数据集对比训练速度
    
    这个测试使用10MB的数据和2048的词汇表大小，
    对比RustBPE和HuggingFace的训练速度。
    
    注意：
        - 标记为@pytest.mark.slow，需要显式运行
        - 优化的Python版本已注释掉（太慢了）
    
    参数：
        enwik8_large: 10MB的enwik8测试数据（来自fixture）
    """
    text = enwik8_large
    vocab_size = 2048
    print(f"\nText length: {len(text)}")

    # 注释掉Python优化版本，因为太慢了
    # 在大数据集上，Rust和HuggingFace的实现都比Python快得多
    # print("Training optimized python version...")
    # optimized_python_tokenizer = FastRegexTokenizer()
    # _, optimized_python_train_time = time_function(optimized_python_tokenizer.train, text, vocab_size)
    # print(f"Optimized python train time: {optimized_python_train_time:.4f}s")

    # ========== 训练RustBPE ==========
    print("\nTraining rustbpe...")
    rustbpe_tokenizer = rustbpe.Tokenizer()
    _, rustbpe_train_time = time_function(rustbpe_tokenizer.train_from_iterator, [text], vocab_size)
    print(f"RustBPE train time: {rustbpe_train_time:.4f}s")
    assert rustbpe_train_time > 0, "Training should take some time"

    # ========== 训练HuggingFace ==========
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(HuggingFaceTokenizer.train_from_iterator, [text], vocab_size)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    assert hf_train_time > 0, "Training should take some time"

    # ========== 打印性能对比 ==========
    print(f"\n📊 Performance comparison:")
    print(f"   RustBPE: {rustbpe_train_time:.4f}s")
    print(f"   HuggingFace: {hf_train_time:.4f}s")
    print(f"   Speedup: {hf_train_time/rustbpe_train_time:.2f}x")

def test_interface(enwik8_small):
    """
    接口测试：测试RustBPETokenizer的完整接口
    
    测试内容：
        1. 训练分词器
        2. 编码/解码文本（包括Unicode）
        3. 批量编码
        4. 特殊token的添加（prepend/append）
        5. 保存和加载分词器
    
    这个测试验证nanochat.tokenizer.RustBPETokenizer包装类的完整功能。
    
    参数：
        enwik8_small: 100KB的enwik8测试数据（来自fixture）
    """
    import tempfile
    from nanochat.tokenizer import RustBPETokenizer

    # ========== 测试1：训练分词器 ==========
    vocab_size = 300
    tok = RustBPETokenizer.train_from_iterator([enwik8_small], vocab_size)
    assert tok.get_vocab_size() == vocab_size, f"Expected vocab size {vocab_size}, got {tok.get_vocab_size()}"
    print(f"✅ Trained tokenizer with vocab size {vocab_size}")

    # ========== 测试2：编码/解码（包括emoji） ==========
    encode_text = "Hello world! How are you? 🙃"
    ids = tok.encode(encode_text)
    print(f"\nInput text: {encode_text}")
    print(f"IDs: {ids}")
    decoded = tok.decode(ids)
    print(f"Decoded: {decoded}")
    assert decoded == encode_text, f"Decoded text doesn't match: {decoded} != {encode_text}"
    print("✅ Encode/decode test passed")

    # ========== 测试3：批量编码 ==========
    ids_new = tok.encode([encode_text, encode_text])
    assert all(x == ids for x in ids_new), "Batch encoding should produce identical results"
    print("✅ Encode batch OK")

    # ========== 测试4：特殊token添加（prepend/append） ==========
    ids_special = tok.encode(encode_text, prepend="<|bos|>", append="<|bos|>")
    bos_token_id = tok.encode_special("<|bos|>")
    assert ids_special == [bos_token_id] + ids + [bos_token_id], "Special tokens not correctly added"
    print("✅ append/prepend OK")

    # ========== 测试5：保存和加载 ==========
    with tempfile.TemporaryDirectory() as tmp_dir:
        tok.save(tmp_dir)
        tok_reloaded = RustBPETokenizer.from_directory(tmp_dir)
        ids_reloaded = tok_reloaded.encode(encode_text)
        assert ids_reloaded == ids, "Reloaded tokenizer should produce same results"
        print("✅ Save/load through temporary directory OK")
