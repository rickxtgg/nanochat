"""
分布式AdamW优化器

来源：
    改编自modded-nanogpt项目，由Keller、@vagrawal等人开发
    
注意：
    这不是一个通用的优化器！它是专门为我们的特定用例设计的，
    使用了ZeRO-2风格的优化器状态分片和梯度归约。

技术背景：
    ZeRO-2（Zero Redundancy Optimizer Stage 2）是一种内存优化技术，
    通过在多个设备之间分片优化器状态来减少每个设备的内存占用。
    这对于训练大型模型至关重要。
"""
import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    分布式AdamW优化器
    
    架构：
        采用ZeRO-2风格，即分片的优化器状态和梯度归约
        
    关键特性：
        - 梯度通过reduce-scatter操作进行平均和分片
        - 每个rank只维护参数的一部分的优化器状态
        - 参数更新后通过all-gather同步到所有rank
        - 支持per-parameter学习率和权重衰减倍数
        
    内存优势：
        相比传统DDP，每个rank的优化器状态内存占用减少到1/world_size
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        """
        初始化分布式AdamW优化器
        
        参数：
            param_groups: 参数组列表
            lr: 学习率（默认1e-3）
            betas: Adam的两个beta参数（默认(0.9, 0.999)）
            eps: 数值稳定性的小常数（默认1e-8）
            weight_decay: 权重衰减系数（默认0.01）
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile  # JIT编译以提高性能
    @torch.no_grad()  # 禁用梯度计算
    def step(self):
        """
        执行单步优化更新
        
        算法流程：
            1. 异步reduce-scatter梯度（平均并分片到各rank）
            2. 等待梯度就绪后，在本rank的参数切片上执行AdamW更新
            3. 异步all-gather更新后的参数（同步到所有rank）
            4. 等待所有通信完成
        
        ZeRO-2优化：
            - 每个rank只存储和更新1/world_size的优化器状态
            - 使用异步通信与计算重叠以提高效率
            - 梯度通信使用reduce-scatter（比all-reduce更高效）
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []  # 存储异步梯度归约的future
        all_reduce_futures: list[torch.Future] = []  # 存储异步参数同步的future
        grad_slices = []  # 存储本rank的梯度切片
        
        # ========== 阶段1：启动异步梯度reduce-scatter ==========
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size  # 每个rank负责的切片大小
                grad_slice = torch.empty_like(grad[:rank_size])  # 为本rank的切片分配空间
                # reduce-scatter: 对梯度求平均，每个rank得到不同的切片
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        # ========== 阶段2：等待梯度就绪，执行AdamW更新 ==========
        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            
            for base in range(len(params)):
                # 等待本参数的梯度切片就绪
                reduce_scatter_futures[idx].wait()
                
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]  # 本rank负责的参数切片
                
                # 获取有效学习率（支持per-parameter学习率倍数）
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                
                # ===== 优化器状态初始化 =====
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)  # 一阶矩估计
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)  # 二阶矩估计
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                # ===== AdamW权重衰减（解耦权重衰减） =====
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                
                # ===== 更新移动平均 =====
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)  # m_t = β1*m_{t-1} + (1-β1)*g_t
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)  # v_t = β2*v_{t-1} + (1-β2)*g_t^2
                
                # ===== 偏差修正 =====
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                
                # ===== 计算更新步长 =====
                denom = exp_avg_sq.sqrt().add_(eps)  # sqrt(v_t) + ε
                step_size = lr * (torch.sqrt(bias2) / bias1)  # 修正后的学习率
                update = exp_avg.div(denom).mul_(step_size)  # step_size * m_t / (sqrt(v_t) + ε)
                p_slice.add_(other=update, alpha=-1.0)  # θ_t = θ_{t-1} - update
                
                idx += 1
                
                # 启动异步all-gather，将更新后的参数切片同步到所有rank
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        
        # ========== 阶段3：等待所有参数同步完成 ==========
        torch.futures.collect_all(all_reduce_futures).wait()
