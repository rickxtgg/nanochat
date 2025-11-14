"""
nanochat通用工具模块

这个模块包含在整个nanochat项目中使用的通用实用函数。

主要功能分类：
    - 日志和输出：彩色日志格式化、print0、banner打印
    - 目录管理：获取基础目录、文件下载（带锁）
    - 分布式训练：DDP检测、获取分布式信息、设备自动检测
    - 计算初始化：设备设置、随机种子、精度配置、DDP初始化
    - 虚拟对象：DummyWandb（用于禁用WandB时）
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

# =============================================================================
# 日志格式化和设置
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    自定义日志格式化器，为日志消息添加ANSI颜色
    
    颜色方案：
        - DEBUG: 青色
        - INFO: 绿色
        - WARNING: 黄色
        - ERROR: 红色
        - CRITICAL: 洋红色
    
    特殊高亮：
        - INFO级别的数字和单位（GB、MB、%、docs）会加粗
        - Shard编号会高亮显示
    """
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 洋红色
    }
    RESET = '\033[0m'  # 重置颜色
    BOLD = '\033[1m'   # 加粗
    
    def format(self, record):
        """格式化日志记录并添加颜色"""
        # 为日志级别名称添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        
        # 格式化消息
        message = super().format(record)
        
        # 为消息的特定部分添加颜色
        if levelname == 'INFO':
            # 高亮数字和百分比
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        
        return message


def setup_default_logging():
    """
    设置默认日志配置
    
    配置：
        - 日志级别: INFO
        - 格式: 彩色格式化
        - 输出: 标准错误流（stderr）
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

# =============================================================================
# 目录管理
# =============================================================================

def get_base_dir():
    """
    获取nanochat的基础目录
    
    目录选择策略：
        1. 如果设置了环境变量NANOCHAT_BASE_DIR，使用它
        2. 否则使用~/.cache/nanochat（默认）
    
    返回：
        nanochat_dir: 基础目录的完整路径
    
    目的：
        将nanochat的中间文件（检查点、数据集、评估数据等）
        与其他缓存数据共同存放在~/.cache中
    """
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    带文件锁的下载函数
    
    参数：
        url: 下载URL
        filename: 保存的文件名（保存在基础目录中）
        postprocess_fn: 可选的后处理函数（如解压）
    
    返回：
        file_path: 下载文件的完整路径
    
    特性：
        - 使用文件锁防止多进程/多rank并发下载
        - 如果文件已存在，直接返回
        - 支持后处理（如解压zip文件）
    
    工作流程：
        1. 检查文件是否已存在
        2. 获取文件锁（只有一个rank能获取）
        3. 再次检查文件（防止在等待锁期间其他rank已下载）
        4. 下载文件
        5. 执行后处理（如果提供）
        6. 释放锁
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    # 快速路径：文件已存在
    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # 只有一个rank能获取这个锁
        # 所有其他rank会阻塞直到锁被释放

        # 获取锁后再次检查（可能在等待期间已被下载）
        if os.path.exists(file_path):
            return file_path

        # 下载内容（字节形式）
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()  # 字节

        # 写入本地文件
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # 如果提供了后处理函数，运行它
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

# =============================================================================
# 输出工具
# =============================================================================

def print0(s="", **kwargs):
    """
    只在rank 0（主进程）打印
    
    参数：
        s: 要打印的字符串（默认空字符串）
        **kwargs: 传递给print函数的其他参数
    
    用途：
        在分布式训练中避免重复打印相同信息。
        只有rank 0（主进程）会打印输出。
    
    实现：
        通过检查环境变量RANK来判断是否为主进程。
        如果RANK未设置，默认为0（非分布式模式）。
    
    示例：
        >>> print0("模型已加载")  # 只在主进程打印
        >>> print0(f"损失: {loss:.4f}", flush=True)  # 支持额外参数
    """
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    """
    打印nanochat的ASCII艺术banner
    
    使用DOS Rebel字体制作（https://manytools.org/hacker-tools/ascii-banner/）
    只在rank 0打印
    """
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

# =============================================================================
# 分布式训练工具
# =============================================================================

def is_ddp():
    """
    检测是否在分布式数据并行（DDP）模式下运行
    
    返回：
        bool: 如果设置了RANK环境变量（且不是-1），返回True
    
    注意：
        这是一个简单的检测方式，可能不是最规范的方法
    """
    return int(os.environ.get('RANK', -1)) != -1


def get_dist_info():
    """
    获取分布式训练信息
    
    返回：
        (ddp, ddp_rank, ddp_local_rank, ddp_world_size): 元组
            - ddp: 是否使用DDP（bool）
            - ddp_rank: 全局rank（0-based）
            - ddp_local_rank: 本地rank（节点内的rank，0-based）
            - ddp_world_size: 总进程数
    
    非DDP模式：
        返回(False, 0, 0, 1)
    
    DDP模式：
        从环境变量读取RANK、LOCAL_RANK、WORLD_SIZE
    """
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def autodetect_device_type():
    """
    自动检测可用的设备类型
    
    优先级：
        1. CUDA（如果可用）
        2. MPS（Macbook Metal Performance Shaders，如果可用）
        3. CPU（备选）
    
    返回：
        device_type: "cuda"、"mps"或"cpu"
    """
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"自动检测到设备类型: {device_type}")
    return device_type

# =============================================================================
# 计算环境初始化
# =============================================================================

def compute_init(device_type="cuda"):
    """
    计算环境的基础初始化
    
    这是一个通用的初始化函数，封装了重复的设置代码。
    
    参数：
        device_type: 设备类型，必须是"cuda"、"mps"或"cpu"之一
    
    返回：
        (ddp, ddp_rank, ddp_local_rank, ddp_world_size, device): 元组
            - ddp: 是否使用DDP（bool）
            - ddp_rank: 全局rank
            - ddp_local_rank: 本地rank
            - ddp_world_size: 总进程数
            - device: torch.device对象
    
    初始化内容：
        1. 设备可用性验证
        2. 随机种子设置（用于可复现性）
        3. 精度配置（CUDA使用TF32）
        4. 分布式训练设置（如果启用DDP）
    """
    # 验证设备类型
    assert device_type in ["cuda", "mps", "cpu"], f"无效的设备类型: {device_type}（必须是'cuda'、'mps'或'cpu'）"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "您的PyTorch安装未配置CUDA支持，但device_type设置为'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "您的PyTorch安装未配置MPS支持，但device_type设置为'mps'"

    # ===== 可复现性设置 =====
    # 注意：我们在这里设置全局随机种子，但大部分代码使用显式的rng对象。
    # 唯一可能使用全局rng的地方是nn.Module的模型权重初始化。
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # 暂时跳过完全确定性算法，因为可能会降低速度，稍后再研究
    # torch.use_deterministic_algorithms(True)

    # ===== 精度设置 =====
    if device_type == "cuda":
        # 使用TF32而非FP32进行矩阵乘法（更快但略微降低精度）
        torch.set_float32_matmul_precision("high")

    # ===== 分布式设置：Distributed Data Parallel (DDP) =====
    # DDP是可选的，仅支持CUDA
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # 将"cuda"默认指向此设备
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()  # 同步所有进程
    else:
        device = torch.device(device_type)  # mps或cpu

    if ddp_rank == 0:
        logger.info(f"分布式训练进程数: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    """
    计算环境清理
    
    这是compute_init的配套函数，在脚本退出前清理资源。
    
    清理内容：
        - 如果使用DDP，销毁进程组
    """
    if is_ddp():
        dist.destroy_process_group()

# =============================================================================
# 虚拟对象
# =============================================================================

class DummyWandb:
    """
    虚拟WandB对象（占位符）
    
    用途：
        当我们不想使用WandB但希望保持相同的函数签名时使用。
        所有方法都是空操作（no-op），不会产生任何副作用。
    
    使用场景：
        1. 禁用WandB日志记录（run="dummy"）
        2. 非主进程不需要记录日志
        3. 调试或测试时不想产生WandB日志
    
    方法：
        - __init__(): 空操作初始化
        - log(): 空操作日志记录
        - finish(): 空操作完成
    
    示例：
        >>> wandb_run = DummyWandb() if use_dummy else wandb.init(...)
        >>> wandb_run.log({"loss": 0.5})  # 如果是DummyWandb则什么都不做
        >>> wandb_run.finish()  # 如果是DummyWandb则什么都不做
    """
    def __init__(self):
        """初始化虚拟WandB对象（空操作）"""
        pass
    
    def log(self, *args, **kwargs):
        """
        空操作日志记录
        
        参数：
            *args: 任意位置参数（被忽略）
            **kwargs: 任意关键字参数（被忽略）
        
        返回：
            None
        """
        pass
    
    def finish(self):
        """
        空操作完成
        
        用途：
            模拟WandB的finish()方法，但不执行任何操作。
        """
        pass
