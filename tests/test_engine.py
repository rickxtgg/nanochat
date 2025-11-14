"""
Engine类的单元测试

这个测试文件专门测试nanochat引擎模块的核心功能。

运行方式：
    python -m pytest tests/test_engine.py -v

主要测试内容：
    - KV缓存（Key-Value Cache）的动态调整大小功能
    - 确保在生成长序列时，缓存能正确扩展而不丢失数据

技术背景：
    在Transformer模型的推理过程中，KV缓存用于存储之前计算过的注意力键和值，
    避免重复计算以提高生成效率。当生成序列超过初始缓存大小时，
    需要动态扩展缓存并保持原有数据完整性。
"""

# PyTorch深度学习框架
import torch
# 导入KV缓存类
from nanochat.engine import KVCache

def test_kv_cache_resize():
    """
    KV缓存动态调整大小测试
    
    测试目标：
        验证KV缓存在序列长度超过初始容量时，能够正确地扩展缓存大小，
        并且保持原有token的键值对数据完整无损。
    
    背景：
        这个测试重现了之前发现的一个KV缓存未正确调整大小的问题。
        详细信息：https://github.com/karpathy/nanochat/pull/186
    
    测试步骤：
        1. 创建一个初始seq_len=4的KV缓存
        2. 插入4个token，填满初始容量
        3. 插入第5个token，触发缓存扩展
        4. 验证缓存确实扩展了
        5. 验证前4个token的数据在扩展后仍然完整
    """

    # 测试配置参数
    batch_size = 2    # 批次大小
    num_heads = 3     # 注意力头数
    seq_len = 4       # 初始序列长度（故意设小以触发扩展）
    head_dim = 5      # 每个注意力头的维度
    num_layers = 6    # 模型层数

    # 创建KV缓存实例
    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers
    )

    # ============= 辅助函数：插入单个token =============
    def insert_token(token_idx):
        """
        向所有层插入一个token的key和value
        
        使用不同的填充值以便后续验证：
        - key使用token_idx作为填充值
        - value使用token_idx * 100作为填充值
        """
        for layer_idx in range(num_layers):
            k = torch.full((batch_size, num_heads, 1, head_dim), fill_value=float(token_idx), dtype=torch.float32)
            v = torch.full((batch_size, num_heads, 1, head_dim), fill_value=float(token_idx * 100), dtype=torch.float32)
            kv_cache.insert_kv(layer_idx, k, v)

    # ============= 步骤1：插入4个token，填满初始缓存 =============
    for i in range(4):
        insert_token(i)

    # ============= 步骤2：记录原始缓存状态 =============
    original_cache = kv_cache.kv_cache.clone()  # 克隆缓存以供后续对比
    original_seq_len = original_cache.shape[4]  # 记录原始序列长度

    # ============= 步骤3：插入第5个token，触发缓存扩展 =============
    insert_token(4)
    
    # 验证缓存确实扩展了
    new_seq_len = kv_cache.kv_cache.shape[4]
    assert new_seq_len > original_seq_len, f"缓存未扩展: original seq_len={original_seq_len}, new seq_len={new_seq_len}"

    # ============= 步骤4：验证前4个token的数据完整性 =============
    for layer_idx in range(num_layers):
        for token_idx in range(4):
            # 计算预期值
            expected_k = float(token_idx)        # key的预期值
            expected_v = float(token_idx * 100)  # value的预期值
            
            # 从扩展后的缓存中获取实际值
            actual_k = kv_cache.kv_cache[layer_idx, 0, :, :, token_idx, :]
            actual_v = kv_cache.kv_cache[layer_idx, 1, :, :, token_idx, :]
            
            # 验证扩展后的缓存与预期值匹配
            assert (actual_k == expected_k).all(), f"Layer {layer_idx}, token {token_idx}: key损坏, 预期 {expected_k}"
            assert (actual_v == expected_v).all(), f"Layer {layer_idx}, token {token_idx}: value损坏, 预期 {expected_v}"
            
            # 验证扩展后的缓存与原始缓存匹配
            original_k = original_cache[layer_idx, 0, :, :, token_idx, :]
            original_v = original_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == original_k).all(), f"Layer {layer_idx}, token {token_idx}: key与原始缓存不匹配"
            assert (actual_v == original_v).all(), f"Layer {layer_idx}, token {token_idx}: value与原始缓存不匹配"
