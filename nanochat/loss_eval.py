"""
基础模型评估辅助函数

本模块提供了用于评估语言模型性能的工具函数，特别是计算bits per byte (BPB)指标。

核心功能：
    - evaluate_bpb: 计算模型的bits per byte指标
    
关键特性：
    - 与词汇表大小无关的评估指标（BPB）
    - 正确处理特殊token（不计入指标）
    - 正确处理masked token（ignore_index=-1）
    - 支持分布式评估（多GPU聚合）
    - 按token的字节长度归一化损失
"""
import math  # 数学函数（log）
import torch  # PyTorch核心
import torch.distributed as dist  # 分布式训练支持

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    计算模型的bits per byte (BPB)指标
    
    参数：
        model: 语言模型实例
        batches: 数据批次迭代器（生成(x, y)对）
        steps: 评估步数
        token_bytes: 每个token的字节长度张量，形状(vocab_size,)
            - 正常token：对应的字节数（1-4字节，取决于UTF-8编码）
            - 特殊token：设置为0（不计入指标）
    
    返回：
        bpb: bits per byte指标（float）
    
    BPB vs 平均损失：
        平均损失的问题：
            - 与词汇表大小相关（改变词汇表大小后无法比较）
            - 不反映实际数据压缩能力
        
        BPB的优势：
            - 与词汇表大小无关（可以比较不同tokenizer）
            - 衡量模型对实际字节的预测能力
            - 更符合信息论原理（压缩率）
    
    计算方法：
        1. 计算总损失（nats）：sum(loss * mask)
        2. 计算总字节数：sum(token_bytes[y] * mask)
        3. BPB = total_nats / (log(2) * total_bytes)
    
    关键特性：
        1. 所有"正常"token按其字节长度归一化
        2. 特殊token（如<|bos|>）被排除（token_bytes=0）
        3. masked token（ignore_index=-1）被排除
        4. 支持分布式：跨所有rank聚合统计
    
    实现细节：
        - 使用loss_reduction='none'获取逐token损失
        - 处理MPS不支持int64负数比较的问题
        - 快速路径：无masked token时直接索引
        - 慢速路径：有masked token时使用torch.where
    """
    # 初始化累加器（在模型设备上）
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())  # 总损失（自然对数）
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())  # 总字节数
    
    # 遍历所有批次，累积损失和字节数
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)  # 获取输入和目标
        
        # 计算逐token损失（不要平均，保留每个token的损失）
        loss2d = model(x, y, loss_reduction='none')  # (B, T)
        loss2d = loss2d.view(-1)  # 展平为一维
        y = y.view(-1)  # 展平目标
        
        # 检查是否有masked token（ignore_index=-1）
        if (y.int() < 0).any():  # MPS设备不支持int64的负数比较，需要转int32
            # 复杂路径：有ignore_index token
            # 任何<0的目标token都要忽略，不能用负数索引token_bytes
            valid = y >= 0  # 有效token的mask
            y_safe = torch.where(valid, y, torch.zeros_like(y))  # 将负索引替换为0（安全）
            
            # 映射有效token到它们的字节长度；被忽略的token贡献0字节
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],  # 有效token：查找字节数
                torch.zeros_like(y, dtype=token_bytes.dtype)  # 无效token：0字节
            )
            
            # 累加损失和字节数（只计算字节数>0的token）
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # 快速路径：没有ignore_index token，可以直接索引
            num_bytes2d = token_bytes[y]  # 查找每个token的字节数
            total_nats += (loss2d * (num_bytes2d > 0)).sum()  # 只累加字节数>0的损失
            total_bytes += num_bytes2d.sum()
    
    # 分布式：跨所有rank聚合统计
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)  # 求和所有rank的损失
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)  # 求和所有rank的字节数
    
    # 转换到CPU并计算BPB
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    
    # 边界情况：没有有效字节（避免除零）
    if total_bytes == 0:
        return float('inf')
    
    # 计算BPB：nats转换为bits（除以log(2)），然后除以总字节数
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
