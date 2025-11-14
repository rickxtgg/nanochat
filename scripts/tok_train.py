"""
分词器训练脚本 - 使用HuggingFace Tokenizers库训练BPE分词器

功能说明：
本脚本训练一个Byte Pair Encoding (BPE) 分词器，类似于GPT-4的分词器风格。
分词器是语言模型的核心组件，负责将文本转换为token序列。

BPE算法原理：
1. 从字符级别开始
2. 迭代合并最频繁的字符对
3. 逐步构建词汇表，直到达到目标大小

训练过程：
1. 从FinewebEdu数据集加载训练文本
2. 限制每个文档的最大长度（避免超长文档）
3. 限制总训练字符数（控制训练时间）
4. 使用BPE算法训练分词器
5. 保存分词器模型
6. 计算并缓存token字节映射（用于BPB评估）

运行方式：

1. 默认参数（10B字符，词汇表大小65536）：
   python -m scripts.tok_train
   
2. 自定义参数：
   python -m scripts.tok_train --max_chars 5000000000 --vocab_size 32768
   说明：使用5B字符训练，词汇表大小32768

3. 快速测试（小词汇表）：
   python -m scripts.tok_train --max_chars 100000000 --vocab_size 16384
   说明：使用100M字符快速训练一个小词汇表

技术特性：
- Rust实现的BPE：使用rustbpe库，性能极高
- 文档长度限制：避免超长文档影响训练
- Token字节映射：用于计算bits per byte (BPB)指标
- 特殊token处理：正确处理<|bos|>、<|user_start|>等特殊token
- UTF-8兼容：正确处理多字节字符（如中文、emoji）
"""
import os  # 操作系统接口
import time  # 时间测量
import argparse  # 命令行参数解析
import torch  # PyTorch（用于保存token字节映射）
from nanochat.tokenizer import RustBPETokenizer  # Rust实现的BPE分词器
from nanochat.common import get_base_dir  # 获取基础目录
from nanochat.dataset import parquets_iter_batched  # Parquet数据集迭代器

# =============================================================================
# 解析命令行参数
# =============================================================================

parser = argparse.ArgumentParser(description='训练BPE分词器')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='最多训练的字符数（默认：10B）')
parser.add_argument('--doc_cap', type=int, default=10_000, help='每个文档的最大字符数（默认：10,000）')
parser.add_argument('--vocab_size', type=int, default=65536, help='词汇表大小（默认：65536 = 2^16）')
args = parser.parse_args()
print(f"最大字符数: {args.max_chars:,}")
print(f"文档字符上限: {args.doc_cap:,}")
print(f"词汇表大小: {args.vocab_size:,}")

# =============================================================================
# 文本迭代器
# =============================================================================

def text_iterator():
    """
    训练文本迭代器
    
    处理步骤：
    1) 将批次展平为单个迭代器
    2) 裁剪每个文档到args.doc_cap个字符（避免超长文档）
    3) 当达到args.max_chars个字符时停止
    
    为什么要限制文档长度？
    - 超长文档会导致BPE训练效率低下
    - 对于大多数token合并，文档前10K字符已经足够代表性
    
    为什么要限制总字符数？
    - 控制训练时间
    - 通常10B字符已经足够训练一个高质量的分词器
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            # 裁剪文档到最大长度
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            # 达到最大字符数后停止
            if nchars > args.max_chars:
                return

text_iter = text_iterator()

# =============================================================================
# 训练分词器
# =============================================================================
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"训练时间: {train_time:.2f}秒")

# =============================================================================
# 保存分词器到磁盘
# =============================================================================
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# =============================================================================
# 快速内联健全性检查
# =============================================================================
# 测试各种类型的文本：普通文本、数字、缩写、特殊字符、Unicode
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, "分词器编码解码测试失败"

# =============================================================================
# 计算并缓存Token字节映射
# =============================================================================
# 为什么需要token字节映射？
# 为了高效评估bits per byte (BPB)。与典型的平均loss不同，
# 这允许我们报告一个与分词器词汇表大小无关的损失。
# 验证集上的bits per byte是我们关心的主要指标之一。

vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []

for token_id in range(vocab_size):
    token_str = token_strings[token_id]  # 此token的Python字符串表示
    if token_str in special_set:
        token_bytes.append(0)  # 特殊字符不计入字节数
    else:
        id_bytes = len(token_str.encode("utf-8"))  # 组成此token的字节数
        token_bytes.append(id_bytes)

# 保存为PyTorch tensor
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"已保存token字节映射到 {token_bytes_path}")

# =============================================================================
# 记录到实验报告
# =============================================================================
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args),  # 命令行参数
    {"train_time": train_time},  # 训练时间
    {"num_special_tokens": len(special_set)},  # 特殊token数量
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
