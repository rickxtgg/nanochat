"""
分布式分词数据加载器

这个模块提供了一个高效的数据加载器，用于预训练阶段。
它从Parquet文件流式读取文本，进行分词，并生成训练批次。

核心特性：
    - 流式读取：无需将整个数据集加载到内存
    - 分布式支持：自动在多个GPU间分配数据
    - 异步传输：使用pinned memory加速CPU到GPU的数据传输
    - 批量分词：支持多线程并行分词以提高效率
    - 无限迭代：自动循环遍历数据集

性能优化：
    - Token缓冲区（deque）用于高效的流式处理
    - 可配置的分词批次大小以平衡内存和速度
    - 非阻塞GPU传输以重叠计算和通信
"""
from collections import deque

import torch

from nanochat.common import get_dist_info
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda"):
    """
    分布式分词数据加载器
    
    从Parquet文件流式读取预训练文本，进行分词，并生成训练批次。
    
    参数：
        B: 批次大小（batch size）
        T: 序列长度（sequence length）
        split: 数据分割，"train"或"val"
        tokenizer_threads: 分词器线程数（默认4）
        tokenizer_batch_size: 分词批次大小（默认128）
        device: 目标设备（默认"cuda"）
    
    生成：
        (inputs, targets): 元组
            - inputs: 形状为(B, T)的输入token张量（int32）
            - targets: 形状为(B, T)的目标token张量（int64）
    
    工作流程：
        1. 从Parquet文件读取文档批次（按DDP rank分配）
        2. 将文档分词并添加BOS token
        3. 将token累积到缓冲区
        4. 当缓冲区有足够token时，构造训练批次
        5. 将批次异步传输到GPU
        6. 无限循环（适用于训练）
    
    分布式：
        - 每个rank读取不同的Parquet行组
        - 使用start=ddp_rank, step=ddp_world_size跨rank分片
    
    内存优化：
        - 使用deque作为token缓冲区（O(1)的pop和append）
        - 使用pinned memory加速CPU到GPU传输
        - 非阻塞传输允许计算和通信重叠
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1  # +1是因为我们还需要最后一个token的目标
    # 获取分词器和BOS token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # Token缓冲区保存一次迭代的token
    token_buffer = deque()  # 我们从右侧流式添加token，从左侧pop

    # 文档批次的无限迭代器
    def document_batches():
        """无限循环遍历数据集的文档批次"""
        while True:
            # 批次将以parquet文件的组大小迭代，通常例如1024行
            for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                # 对于分词器，我们可能想要使用通常更小的批次，例如128行
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()

    batch_index = 0
    while True:
        # 在生成之前累积足够的token
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
        
        # 将token从deque移动到临时缓冲区
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA支持内存固定以加快CPU和GPU之间的传输：
        scratch = torch.tensor(tokens, dtype=torch.int64, pin_memory=(device == "cuda"))
        # 创建输入/目标作为1D张量
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        # 重塑为2D并异步移动到GPU
        inputs = inputs_cpu.view(B, T).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(B, T).to(device=device, dtype=torch.int64, non_blocking=True)
        yield inputs, targets
