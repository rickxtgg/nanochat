"""
GPT模型实现（重写版，更简洁）

本模块实现了一个现代化的GPT Transformer模型，采用多项最新技术。

显著特性：
    1. Rotary Position Embeddings (RoPE)：
       - 旋转位置编码，无需学习的位置嵌入
       - 更好的长度外推能力
    
    2. QK Normalization：
       - 对Query和Key进行RMSNorm归一化
       - 提高训练稳定性
    
    3. Untied Weights：
       - token embedding和lm_head权重不共享
       - 更灵活的表示能力
    
    4. ReLU² Activation：
       - MLP中使用ReLU²（ReLU平方）激活函数
       - 更强的非线性表达能力
    
    5. Norm After Embedding：
       - token embedding后立即归一化
       - 稳定训练初期
    
    6. Pure Functional RMSNorm：
       - RMSNorm无可学习参数
       - 减少参数量，加速训练
    
    7. No Bias：
       - 所有Linear层无bias
       - 简化架构，减少参数
    
    8. Multi-Query Attention (MQA)：
       - 支持不同数量的Query和KV头
       - 更高效的推理（KV cache更小）

架构组件：
    - GPTConfig: 模型配置（超参数）
    - CausalSelfAttention: 因果自注意力层（支持MQA/GQA）
    - MLP: 前馈网络（ReLU²激活）
    - Block: Transformer块（Attention + MLP + Pre-norm）
    - GPT: 完整模型（Embedding + Blocks + LM Head）
"""

import math  # 数学函数
from functools import partial  # 偏函数
from dataclasses import dataclass  # 数据类装饰器

import torch  # PyTorch核心
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数

from nanochat.common import get_dist_info, print0  # 分布式工具
from nanochat.muon import Muon, DistMuon  # Muon优化器
from nanochat.adamw import DistAdamW  # 分布式AdamW

@dataclass
class GPTConfig:
    """
    GPT模型配置
    
    参数：
        sequence_len: 序列最大长度（默认1024）
        vocab_size: 词汇表大小（默认50304，是64的倍数以优化性能）
        n_layer: Transformer层数（默认12）
        n_head: Query头数（默认6）
        n_kv_head: Key/Value头数（默认6）
            - 等于n_head：标准Multi-Head Attention (MHA)
            - 小于n_head：Multi-Query Attention (MQA) 或 Grouped-Query Attention (GQA)
            - MQA/GQA可以减少KV cache大小，加速推理
        n_embd: 嵌入维度/模型维度（默认768）
    
    注意：
        - n_embd必须能被n_head整除（head_dim = n_embd // n_head）
        - n_head必须能被n_kv_head整除（每个KV头服务多个Query头）
    """
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # Query头数
    n_kv_head: int = 6  # Key/Value头数（MQA/GQA）
    n_embd: int = 768


def norm(x):
    """
    纯函数式RMSNorm（无可学习参数）
    
    参数：
        x: 输入张量，形状(..., d)
    
    返回：
        归一化后的张量，形状与输入相同
    
    RMSNorm公式：
        y = x / RMS(x) = x / sqrt(mean(x²) + eps)
        
    优势：
        - 无可学习参数（简化架构）
        - 计算高效（比LayerNorm少一个减法）
        - 训练稳定性好
    """
    # PyTorch内置的RMSNorm实现（高效）
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """
    应用旋转位置编码（Rotary Position Embedding, RoPE）
    
    参数：
        x: 输入张量，形状(B, H, T, D)（多头注意力格式）
        cos: 余弦表，形状(1, T, 1, D/2)
        sin: 正弦表，形状(1, T, 1, D/2)
    
    返回：
        旋转后的张量，形状(B, H, T, D)
    
    RoPE原理：
        将每对相邻维度视为一个复数，进行旋转：
        [x1, x2] -> [x1*cos + x2*sin, x1*(-sin) + x2*cos]
        
        旋转角度与位置t成正比，实现相对位置编码
    
    优势：
        - 无需学习参数
        - 长度外推能力强
        - 捕获相对位置信息
    """
    assert x.ndim == 4, "期望4D张量（多头注意力格式）"  # (B, H, T, D)
    
    d = x.shape[3] // 2  # 每对维度的大小
    x1, x2 = x[..., :d], x[..., d:]  # 分割最后一维为两半
    
    # 旋转每对维度（复数旋转）
    y1 = x1 * cos + x2 * sin  # 实部
    y2 = x1 * (-sin) + x2 * cos  # 虚部
    
    # 重新组装
    out = torch.cat([y1, y2], 3)
    
    # 确保输入/输出dtype匹配（cos/sin可能是bfloat16）
    out = out.to(x.dtype)
    
    return out

class CausalSelfAttention(nn.Module):
    """
    因果自注意力层（支持MQA/GQA）
    
    特性：
        1. Multi-Query/Grouped-Query Attention：
           - 支持不同数量的Query和KV头
           - 减少KV cache大小，提升推理速度
        
        2. Rotary Position Embeddings：
           - 对Q和K应用RoPE
           - 实现相对位置编码
        
        3. QK Normalization：
           - 对Q和K进行RMSNorm
           - 提高训练稳定性
        
        4. 因果Masking：
           - 自回归生成（每个token只能看到之前的token）
           - 支持KV cache（推理优化）
    
    前向传播流程：
        1. 线性投影：x -> Q, K, V
        2. 应用RoPE：Q, K -> Q', K'
        3. QK归一化：norm(Q'), norm(K')
        4. 重塑为多头：(B, T, D) -> (B, H, T, D/H)
        5. KV cache插入（如果有）
        6. Scaled Dot-Product Attention（带因果mask）
        7. 重塑回：(B, H, T, D/H) -> (B, T, D)
        8. 输出投影：x -> x
    """
    def __init__(self, config, layer_idx):
        """
        初始化因果自注意力层
        
        参数：
            config: GPTConfig配置对象
            layer_idx: 层索引（用于KV cache）
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head  # Query头数
        self.n_kv_head = config.n_kv_head  # KV头数
        self.n_embd = config.n_embd  # 嵌入维度
        self.head_dim = self.n_embd // self.n_head  # 每个头的维度
        
        # 验证配置
        assert self.n_embd % self.n_head == 0, "n_embd必须能被n_head整除"
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0, \
            "n_kv_head必须<=n_head且n_head必须能被n_kv_head整除"
        
        # Q, K, V投影层（无bias）
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # 输出投影层
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        """
        前向传播
        
        参数：
            x: 输入张量，形状(B, T, C)
            cos_sin: RoPE的(cos, sin)元组
            kv_cache: KV缓存对象（推理时使用，训练时为None）
        
        返回：
            y: 输出张量，形状(B, T, C)
        """
        B, T, C = x.size()

        # 1. 投影输入以获取Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)  # (B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)  # (B, T, n_kv_head, head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)  # (B, T, n_kv_head, head_dim)

        # 2. 应用Rotary Embeddings到Q和K（获取相对位置编码）
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)  # RoPE
        q, k = norm(q), norm(k)  # QK归一化（提高稳定性）
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, H, T, D)

        # 3. 应用KV cache：插入当前k,v到缓存，获取完整视图
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)  # 当前前向传播的query数量
        Tk = k.size(2)  # 总的key/value数量（缓存+当前）

        # 4. 注意力：queries自回归地attend到keys/values。需要处理几种情况：
        enable_gqa = self.n_head != self.n_kv_head  # GQA：如果需要，复制KV头以匹配Q头
        
        if kv_cache is None or Tq == Tk:
            # 情况1：训练时（无KV cache），使用标准因果注意力
            # 或即使有KV cache，当Tq==Tk时也可以使用这个简单版本
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # 情况2：推理时但只有单个query
            # query需要attend到缓存中的所有keys/values
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # 情况3：推理时且有一组queries
            # 首先，每个query attend到所有缓存的keys/values（完整前缀）
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)  # True=保留, False=mask
            prefix_len = Tk - Tq
            if prefix_len > 0:  # 不能为负但可能为0
                attn_mask[:, :prefix_len] = True
            # 然后，在这组queries内进行因果注意力
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # 5. 重新组装各头并投影回残差流
        y = y.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    多层感知机（MLP）/ 前馈网络（Feed-Forward Network）
    
    架构：
        x -> Linear(d, 4d) -> ReLU² -> Linear(4d, d) -> x
    
    特性：
        1. ReLU²激活：
           - ReLU(x)² = max(0, x)²
           - 比标准ReLU更强的非线性
           - 简单有效
        
        2. 4倍扩展：
           - 隐藏层维度是输入的4倍
           - 标准Transformer配置
        
        3. 无bias：
           - 简化架构
           - 减少参数
    """
    def __init__(self, config):
        """初始化MLP"""
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)  # 上投影
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)  # 下投影

    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入张量，形状(B, T, C)
        
        返回：
            输出张量，形状(B, T, C)
        """
        x = self.c_fc(x)  # (B, T, 4C)
        x = F.relu(x).square()  # ReLU²激活
        x = self.c_proj(x)  # (B, T, C)
        return x


class Block(nn.Module):
    """
    Transformer块（Pre-norm架构）
    
    架构：
        x -> norm -> attention -> + -> norm -> mlp -> + -> output
        |__________________________|    |__________________|
               (残差连接1)                    (残差连接2)
    
    特性：
        1. Pre-normalization：
           - 在attention和mlp之前归一化
           - 更稳定的训练
        
        2. 残差连接：
           - 缓解梯度消失
           - 允许深层网络
        
        3. 纯函数式RMSNorm：
           - 无可学习参数
           - 简化架构
    """
    def __init__(self, config, layer_idx):
        """
        初始化Transformer块
        
        参数：
            config: GPTConfig配置对象
            layer_idx: 层索引
        """
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        """
        前向传播
        
        参数：
            x: 输入张量，形状(B, T, C)
            cos_sin: RoPE的(cos, sin)元组
            kv_cache: KV缓存对象
        
        返回：
            输出张量，形状(B, T, C)
        """
        x = x + self.attn(norm(x), cos_sin, kv_cache)  # 注意力 + 残差
        x = x + self.mlp(norm(x))  # MLP + 残差
        return x


class GPT(nn.Module):
    """
    GPT模型（完整实现）
    
    架构：
        Embedding -> Norm -> Blocks (n_layer) -> Norm -> LM Head
    
    组件：
        - wte: Token嵌入层
        - h: Transformer块列表（n_layer个）
        - lm_head: 语言模型头（输出logits）
        - cos/sin: 预计算的RoPE表（非持久化buffer）
    
    特性：
        1. Untied weights：wte和lm_head权重不共享
        2. Logits softcap：限制logits范围（±15）
        3. 混合优化器：Muon（矩阵）+ AdamW（embedding/lm_head）
        4. Meta device支持：可以在meta device上初始化
    """
    def __init__(self, config):
        """
        初始化GPT模型
        
        参数：
            config: GPTConfig配置对象
        """
        super().__init__()
        self.config = config
        
        # Transformer主体
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),  # Token嵌入
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),  # Transformer块
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 语言模型头
        
        # 预计算RoPE（支持meta device初始化，这里先是fake的）
        # RoPE表很小，内存开销低，所以我们过度计算（10倍），简化实现
        # 未来可以动态增长，目前这样就够用
        self.rotary_seq_len = config.sequence_len * 10  # 10倍过度计算，TODO: 可以更优雅
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # persistent=False：不保存到checkpoint（可以随时重新计算）
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        """
        初始化模型权重
        
        初始化策略：
            1. 应用_init_weights到所有模块（正常初始化）
            2. 零初始化：lm_head、所有block的c_proj（输出投影）
            3. 重新计算RoPE表（覆盖meta device的fake版本）
            4. 将embedding转为bfloat16（节省内存）
        
        零初始化的原因：
            输出投影零初始化有助于训练初期的稳定性
            使残差连接在开始时接近恒等映射
        """
        # 应用标准权重初始化
        self.apply(self._init_weights)
        
        # 零初始化分类器权重
        torch.nn.init.zeros_(self.lm_head.weight)
        
        # 零初始化所有block的输出投影权重
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        
        # 重新计算RoPE（覆盖__init__中的fake版本）
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        
        # 将embedding转为bfloat16：优化器可以容忍，且节省内存（模型+激活）
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        """
        权重初始化辅助函数
        
        参数：
            module: 要初始化的模块
        
        初始化方法：
            - Linear: 自适应标准差（基于fan_in和fan_out）
              参考：https://arxiv.org/pdf/2310.17813
              std = 1/√fan_in * min(1, √(fan_out/fan_in))
            - Embedding: 标准正态分布（mean=0, std=1）
        """
        if isinstance(module, nn.Linear):
            # 自适应初始化（考虑输入输出维度比例）
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: 增加base theta，例如100K最近更常见
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """
        预计算Rotary Position Embeddings
        
        参数：
            seq_len: 序列长度
            head_dim: 每个注意力头的维度
            base: RoPE的基础频率（默认10000）
            device: 目标设备（None=自动检测）
        
        返回：
            (cos, sin): 预计算的余弦和正弦表
            形状：(1, seq_len, 1, head_dim/2)
        
        RoPE公式：
            inv_freq[i] = 1 / (base^(2i/head_dim))
            freqs[t, i] = t * inv_freq[i]
            cos/sin = cos/sin(freqs)
        """
        # 自动检测设备（从模型embedding）
        if device is None:
            device = self.transformer.wte.weight.device
        
        # 计算通道的逆频率（步长为2，因为成对旋转）
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        
        # 计算时间步
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        
        # 计算每个(时间, 通道)对的旋转频率
        freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
        cos, sin = freqs.cos(), freqs.sin()
        
        # 转为bfloat16（节省内存）
        cos, sin = cos.bfloat16(), sin.bfloat16()
        
        # 添加batch和head维度以便后续广播
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
        
        return cos, sin

    def get_device(self):
        """获取模型所在设备"""
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        估算每token的FLOPs
        
        返回：
            num_flops_per_token: 每token的浮点运算次数
        
        参考：
            https://arxiv.org/abs/2204.02311
        
        计算公式：
            FLOPs/token ≈ 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
            - 6x：前向传播（2x）+ 反向传播（4x）
            - 12lhqt：注意力的额外开销
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """
        设置混合优化器（AdamW + Muon）
        
        参数：
            unembedding_lr: lm_head的学习率（默认0.004）
            embedding_lr: embedding的学习率（默认0.2）
            matrix_lr: 矩阵参数（Transformer块）的学习率（默认0.02）
            weight_decay: 权重衰减（默认0.0）
        
        返回：
            [adamw_optimizer, muon_optimizer]: 优化器列表
        
        参数分组策略：
            1. lm_head + embedding -> AdamW（适合1D参数）
            2. Transformer块（所有Linear层）-> Muon（适合2D参数）
        
        学习率缩放：
            LR ∝ 1/√(dmodel/768)
            这样可以在不同模型大小间迁移超参数
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # 将所有参数分为3组（matrix, embedding, lm_head）
        matrix_params = list(self.transformer.h.parameters())  # Transformer块（2D矩阵）
        embedding_params = list(self.transformer.wte.parameters())  # Token embedding（1D）
        lm_head_params = list(self.lm_head.parameters())  # 语言模型头（2D但用AdamW）
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        
        # 为embedding和lm_head创建AdamW优化器
        # 按∝1/√dmodel缩放学习率（超参数是为768维模型调优的）
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"按∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}缩放AdamW参数的学习率")
        
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        # 为线性层创建Muon优化器
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        
        # 合并两个优化器到一个列表
        optimizers = [adamw_optimizer, muon_optimizer]
        
        # 为所有参数组记录初始学习率（用于LR调度）
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """
        前向传播
        
        参数：
            idx: 输入token IDs，形状(B, T)
            targets: 目标token IDs（训练时），形状(B, T)
            kv_cache: KV缓存对象（推理时）
            loss_reduction: 损失聚合方式（'mean'或'none'）
        
        返回：
            - 如果targets不为None：loss（标量或张量）
            - 否则：logits，形状(B, T, vocab_size)
        
        特性：
            1. Logits softcap：限制logits范围到±15
               logits' = 15 * tanh(logits / 15)
            2. 混合精度：logits计算用fp32（数值稳定性）
        """
        B, T = idx.size()

        # 获取当前序列长度的RoPE（形状(1, seq_len, 1, head_dim)）
        assert T <= self.cos.size(1), f"序列长度超出RoPE缓存：{T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"RoPE和idx在不同设备：{idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "RoPE必须是bfloat16"
        
        # 如果有KV cache，需要偏移RoPE到当前位置
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]  # 截取到当前序列长度

        # 前向传播Transformer主体
        x = self.transformer.wte(idx)  # Token embedding
        x = norm(x)  # Embedding后立即归一化
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)  # Transformer块
        x = norm(x)  # 最后归一化

        # 前向传播lm_head（计算logits）
        softcap = 15  # Logits软上限
        if targets is not None:
            # 训练模式：计算并返回损失
            # TODO: 实验Liger Kernels / 分块交叉熵等
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # Logits softcap
            logits = logits.float()  # 使用tf32/fp32（数值稳定）
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), 
                                   ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # 推理模式：计算并返回logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # Logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        简单的自回归流式推理
        
        参数：
            tokens: 初始token列表（Python list）
            max_tokens: 最大生成token数
            temperature: 采样温度（0=贪婪，>1=更随机）
            top_k: Top-K采样（None=禁用）
            seed: 随机种子（可复现）
        
        生成：
            逐个生成的token（int）
        
        简化假设：
            - batch_size = 1
            - 输入/输出都是Python list和int
            - 无KV cache（简单实现，效率较低）
        """
        assert isinstance(tokens, list), "tokens必须是列表"
        device = self.get_device()
        
        # 设置随机数生成器（如果temperature > 0）
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        # 添加batch维度
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, T)
        
        # 自回归生成循环
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # 只取最后一个位置的logits (B, vocab_size)
            
            # Top-K采样（如果启用）
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')  # 屏蔽top-k之外的token
            
            # 采样或贪婪选择
            if temperature > 0:
                logits = logits / temperature  # 温度缩放
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)  # 贪婪
            
            # 拼接新token
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
