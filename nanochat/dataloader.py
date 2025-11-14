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
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    分布式分词数据加载器（支持状态恢复）
    
    从Parquet文件流式读取预训练文本,进行分词,生成训练批次。
    
    参数：
        B: 批次大小
        T: 序列长度
        split: 数据集分割，必须是"train"或"val"
        tokenizer_threads: 分词器线程数（默认4）
        tokenizer_batch_size: 分词批次大小（默认128）
        device: 目标设备（默认"cuda"）
        resume_state_dict: 恢复状态字典（可选）
    
    生成：
        (inputs, targets, state_dict): 元组
            - inputs: 输入token张量，形状(B, T)
            - targets: 目标token张量，形状(B, T)
            - state_dict: 当前状态字典（用于恢复训练）
    
    状态恢复特性：
        - 实现支持近似恢复训练（approximate resume）
        - 返回的state_dict可以传递给resume_state_dict参数以恢复训练
        - 恢复是近似的：不会重复文档但可能跳过少量数据
        - 完美恢复是可能的但会使实现复杂化，当前简化版已足够使用
    
    实现原理：
        1. 无限迭代Parquet文件和行组（支持多轮训练）
        2. DDP模式下每个rank读取不同的行组（避免重复）
        3. 使用deque缓冲区流式处理token
        4. 支持异步GPU传输（CUDA pinned memory）
    """
    assert split in ["train", "val"], "split必须是'train'或'val'"

    # 获取分布式训练信息
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    def document_batches():
        """
        文档批次生成器（无限迭代）
        
        生成：
            (doc_batch, (pq_idx, rg_idx)): 元组
                - doc_batch: 文本列表（来自一个行组）
                - pq_idx: 当前Parquet文件索引
                - rg_idx: 当前行组索引
        
        DDP策略：
            每个rank读取不同的行组（stride = ddp_world_size）
            rank 0读取行组0, world_size, 2*world_size, ...
            rank 1读取行组1, world_size+1, 2*world_size+1, ...
        """
        # 获取所有Parquet文件路径
        parquet_paths = list_parquet_files()
        # 训练集：除最后一个文件外的所有文件；验证集：最后一个文件
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        
        # 从恢复点开始（如果提供）
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        pq_idx = resume_pq_idx  # 从恢复索引开始（默认为0）
        
        while True:  # 无限迭代（多轮训练）
            while pq_idx < len(parquet_paths):  # 遍历所有Parquet文件
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                
                # 确定起始行组索引
                # 如果正在恢复同一文件，从恢复点开始；否则从DDP rank开始
                # 注意：状态恢复有点技巧性，但这是权衡简单性的结果
                if resume_rg_idx is not None:
                    # 计算基准索引（以ddp_world_size为单位）
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1  # 前进1步，确保不重复数据
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None  # 只执行一次，然后重置
                else:
                    rg_idx = ddp_rank  # 从当前rank开始
                
                # 遍历行组（每个rank步长为world_size）
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()  # 每个批次是一个行组，如1024行
                    
                    # 分词器可能需要更小的批次（如128行）
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    
                    rg_idx += ddp_world_size  # 前进到下一个行组（DDP步长）
                
                pq_idx += 1  # 前进到下一个Parquet文件
    
    batches = document_batches()

    # 生成token批次
    needed_tokens = B * T + 1  # +1是因为需要最后一个token的目标
    
    # 获取分词器和BOS token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    
    # token缓冲区（deque用于高效的流式处理）
    token_buffer = deque()  # 从右边添加，从左边弹出
    
    while True:
        # 累积足够的token（至少needed_tokens个）
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            # 批量分词（支持多线程加速）
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            # 将所有token添加到缓冲区
            for tokens in token_lists:
                token_buffer.extend(tokens)
        
        # 从deque中取出需要的token
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        
        # CUDA支持pinned memory，实现CPU到GPU的异步传输
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)  # long=int64
        
        # 创建输入/目标1D张量（目标是输入右移1位）
        inputs_cpu = scratch[:-1]  # 前B*T个token
        targets_cpu = scratch[1:]  # 后B*T个token
        
        # 重塑为2D并异步传输到GPU
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        
        # 保存当前状态（用于近似恢复训练）
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}
        
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    """
    分布式分词数据加载器（简化版，不返回状态）
    
    这是tokenizing_distributed_data_loader_with_state的简化包装函数。
    只返回输入和目标，不返回状态字典。
    
    参数：
        *args, **kwargs: 传递给tokenizing_distributed_data_loader_with_state
    
    生成：
        (inputs, targets): 元组
            - inputs: 输入token张量，形状(B, T)
            - targets: 目标token张量，形状(B, T)
    
    用途：
        当不需要状态恢复功能时使用（如评估、简单训练）
    """
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets