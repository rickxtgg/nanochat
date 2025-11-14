"""
Muon优化器实现

来源：
    - Keller et al. 的Muon优化器论文
    - modded-nanogpt项目的想法借鉴

Muon = MomentUm Orthogonalized by Newton-schulz

核心思想：
    在标准SGD-momentum的基础上，对每个2D参数的更新进行正交化后处理，
    将更新替换为最近的正交矩阵。使用Newton-Schulz迭代高效地正交化更新，
    可以在GPU上稳定地以bfloat16运行。

实现：
    1. Muon: 单机版优化器
    2. DistMuon: 分布式版优化器（支持DDP）
       - 使用reduce_scatter平均梯度
       - 使用all_gather复制更新后的权重

参考资料：
    https://kellerjordan.github.io/posts/muon/
"""
import torch  # PyTorch核心
from torch import Tensor  # 张量类型
import torch.distributed as dist  # 分布式训练支持

@torch.compile  # 编译加速
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    通过Newton-Schulz迭代计算零次幂/正交化
    
    参数：
        G: 输入矩阵（至少2D），形状(..., m, n)
        steps: Newton-Schulz迭代步数（通常5步）
    
    返回：
        X: G的正交化版本，形状与G相同
    
    算法原理：
        使用五次（quintic）迭代，其系数经过选择以最大化零点处的斜率。
        为了最小化迭代步数，经验上有效的做法是继续增加零点处的斜率，
        即使超过了迭代在整个区间上完全收敛到1的点。
    
    输出特性：
        此迭代不产生UV^T，而是产生类似US'V^T的结果，
        其中S'是对角矩阵，S_{ii}' ~ Uniform(0.5, 1.5)。
        这对模型性能没有任何损害（相对于UV^T），其中USV^T=G是SVD。
    
    实现细节：
        1. 支持批量处理（@scottjmaddox的实现）
        2. 自动处理矩阵转置（高>宽时转置）
        3. 谱范数归一化（确保<=1）
        4. 使用bfloat16精度（稳定且高效）
        5. 五次迭代公式：X = aX + (bA + cA²)X，其中A = XX^T
    
    系数来源：
        a, b, c经过优化以最大化收敛速度
        策略改编自@jxbz、@leloykun和@YouJiacheng的建议
    """
    assert G.ndim >= 2, "输入必须至少是2D张量"  # 批量Muon实现
    
    # 优化的五次迭代系数
    a, b, c = (3.4445, -4.7750,  2.0315)
    
    # 转换为bfloat16（稳定且节省内存）
    X = G.bfloat16()
    
    # 如果矩阵更高，转置它（优化性能）
    if G.size(-2) > G.size(-1):
        X = X.mT

    # 确保谱范数最多为1（归一化）
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # 执行Newton-Schulz迭代
    for _ in range(steps):
        A = X @ X.mT  # 计算XX^T
        B = b * A + c * A @ A  # 五次计算：bA + cA²
        X = a * X + B @ X  # 更新：aX + BX

    # 如果之前转置了，现在转置回来
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon优化器（单机版）- 通过Newton-Schulz正交化的动量优化器
    
    论文链接：https://kellerjordan.github.io/posts/muon/
    
    工作原理：
        1. 内部运行标准SGD-momentum
        2. 执行正交化后处理步骤
        3. 将每个2D参数的更新替换为最近的正交矩阵
        4. 使用Newton-Schulz迭代高效正交化（可在GPU上以bfloat16稳定运行）
    
    ⚠️ 使用警告：
        - 不应用于embedding层、最终全连接层或任何{0,1}-D参数
          这些应该使用标准方法优化（如AdamW）
        - 对于4D卷积滤波器，可以将最后3个维度展平使用
    
    适用场景：
        ✅ Transformer的2D线性层（权重矩阵）
        ❌ Embedding层（1D）
        ❌ LayerNorm/RMSNorm参数（0D或1D）
        ❌ 最终分类层（lm_head）
    
    参数：
        params: 可迭代的参数（Tensor）
        lr: 内部SGD使用的学习率（默认0.02）
        momentum: 内部SGD使用的动量系数（默认0.95）
        nesterov: 是否使用Nesterov风格的动量（推荐，默认True）
        ns_steps: Newton-Schulz迭代步数（默认5）
    
    参数分组策略：
        按参数的numel()（元素总数）分组，相同大小的参数在一个组中
        这样可以批量处理相同形状的参数，提高效率
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        
        # 按参数大小分组（相同大小的参数可以批量处理）
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        """
        执行单步优化
        
        算法流程：
            1. 遍历所有参数组
            2. 对每个参数：
               a. 获取梯度g
               b. 更新动量缓冲区：buf = momentum * buf + (1 - momentum) * g
               c. 如果使用Nesterov：g = momentum * buf + (1 - momentum) * g
                  否则：g = buf
               d. 正交化更新：g = zeropower_via_newtonschulz5(g)
               e. 应用更新：p -= lr * scale * g
        
        长宽比缩放：
            scale = sqrt(max(1, height/width))
            这确保了更"高"的矩阵（更多输出特征）获得更大的步长
        """
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None, "所有参数必须有梯度"
                
                # 获取或初始化动量缓冲区
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                
                # 更新动量缓冲区（指数移动平均）
                buf.lerp_(g, 1 - group["momentum"])  # buf = momentum * buf + (1 - momentum) * g
                
                # Nesterov动量或标准动量
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                
                # 正交化更新（Muon的核心）
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                
                # 应用更新（带长宽比缩放）
                scale = max(1, p.size(-2) / p.size(-1))**0.5  # sqrt(height/width)
                p.add_(g, alpha=-group["lr"] * scale)


class DistMuon(torch.optim.Optimizer):
    """
    Muon优化器（分布式版）- 带自主分布式同步的Muon
    
    算法流程：
        1. SGD-momentum（可选Nesterov）
        2. 通过Newton-Schulz正交化2D更新
        3. 应用长宽比缩放的步长
        4. 执行自主的分布式同步：
           - reduce_scatter(AVG)：梯度平均
           - all_gather：复制更新后的权重
    
    分布式策略：
        - 参数按block-cyclic方式分配给不同rank
        - 每个参数有一个"所有者"rank负责计算更新
        - 所有者rank维护动量缓冲区
        - 更新后通过all_gather广播到所有rank
    
    ⚠️ 注意事项：
        1. 仅用于2D参数（如linear/conv核，重塑为2D）
           不要用于0D/1D参数（如embedding或标量）
        
        2. 动量缓冲区仅在"所有者"rank上维护（按block-cyclic分配）
           如果在单个rank上保存优化器状态，请事先合并状态
        
        3. 所有rank必须有相同的参数顺序（排序保证确定性）
    
    参数：
        params: 可迭代的Tensor（所有参数必须是2D）
        lr: 学习率（默认0.02）
        momentum: 动量系数，范围[0,1)（默认0.95）
        nesterov: 是否使用Nesterov风格（默认True）
        ns_steps: Newton-Schulz迭代步数（默认5）
    
    参数分组策略：
        按形状分组（相同形状的参数在一个组）
        这允许批量处理和高效的reduce_scatter/all_gather
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        
        # 验证所有参数都是2D
        assert all(p.ndim == 2 for p in params), "Muon期望仅有2D参数"
        
        rank = dist.get_rank()
        
        # 按形状分组所有参数（排序以确保一致性/确定性）
        shapes = sorted({p.shape for p in params})
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            
            # 验证组内所有参数的device和dtype一致
            assert all(p.device == device for p in group_params), "组内参数device必须相同"
            assert all(p.dtype == dtype for p in group_params), "组内参数dtype必须相同"
            
            # 打印分组信息（仅rank 0）
            if rank == 0:
                print(f"Muon: 分组{len(group_params)}个参数，形状{shape}，设备{device}，类型{dtype}")
            
            # 创建零缓冲区（用于填充）
            param_groups.append(dict(
                params=group_params,
                zero_buffer=torch.zeros_like(group_params[0])
            ))
        
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        """
        执行单步分布式优化
        
        算法流程（两阶段）：
            阶段1：reduce_scatter梯度（异步）
                - 将参数分为world_size大小的组
                - 每个rank"拥有"组中的一个参数（block-cyclic分配）
                - reduce_scatter将每个参数的梯度平均到所有者rank
            
            阶段2：计算更新并all_gather（异步）
                - 所有者rank计算Muon更新（momentum + 正交化）
                - all_gather将更新后的参数广播到所有rank
        
        关键优化：
            - 异步操作：reduce_scatter和all_gather并发执行
            - 流水线：下一组的reduce_scatter在当前组的更新计算时开始
            - 内存效率：仅所有者rank维护动量缓冲区
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 确保所有参数都有梯度
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), \
            "所有参数必须有梯度"

        # ====================
        # 阶段1：启动所有reduce_scatter操作（梯度平均）
        # ====================
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            
            # 以world_size为单位遍历参数
            for base_i in range(0, len(params), world_size):
                # 每个参数的计算所有者是：rank (i % world_size)
                owner_idx = base_i + rank  # 当前rank拥有的参数索引
                
                # 每个rank收集这组world_size个参数的梯度
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                
                # 用零缓冲区填充，使列表长度为world_size
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                
                # 输出缓冲区：所有者rank的梯度（或空缓冲区）
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                
                # reduce_scatter：将world_size个梯度平均到所有者rank
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # ====================
        # 阶段2：每个rank计算更新并收集（流水线式）
        # ====================
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            
            # 以world_size为单位遍历参数
            for base_i in range(0, len(params), world_size):
                # 当前rank拥有的参数索引
                owner_idx = base_i + rank
                
                # 等待reduce_scatter完成（梯度已平均）
                all_reduce_futures[future_idx].wait()  # 未来可以用wait_any轮询优化
                future_idx += 1
                
                # 所有者rank计算Muon更新
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # 现在已经是跨rank平均的梯度
                    
                    # 获取或初始化动量缓冲区
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    
                    # 更新动量缓冲区
                    buf.lerp_(g, 1.0 - group["momentum"])
                    
                    # Nesterov或标准动量
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    
                    # 正交化更新（Muon核心）
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    
                    # 应用更新（带长宽比缩放）
                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)
                
                # 将更新后的参数复制到所有rank
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))])  # 填充
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        # 等待所有工作完成
        torch.futures.collect_all(all_gather_futures).wait()
