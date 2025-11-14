"""
训练报告生成工具

本模块提供用于生成训练报告卡的工具函数和类。

核心功能：
    1. 系统信息收集：Git、GPU、CPU、内存等
    2. 成本估算：基于GPU类型和运行时间
    3. 报告生成：Markdown格式的训练报告
    4. 报告重置：清除旧报告，开始新实验

主要组件：
    - Report类：管理日志和报告生成
    - DummyReport类：用于非rank 0进程的虚拟报告
    - 各种信息收集函数：get_git_info、get_gpu_info、get_system_info等

注意：
    代码比平时更乱一些，将来会修复。
"""

import os  # 文件和目录操作
import re  # 正则表达式
import shutil  # 文件复制
import subprocess  # 执行shell命令
import socket  # 主机名
import datetime  # 时间戳
import platform  # 平台信息
import psutil  # 系统信息（CPU/内存）
import torch  # PyTorch（GPU信息）

def run_command(cmd):
    """
    运行shell命令并返回输出
    
    参数：
        cmd: 要执行的shell命令（字符串）
    
    返回：
        命令输出（字符串），如果失败则返回None
    
    特性：
        - 5秒超时（防止挂起）
        - 捕获所有异常
        - 只返回成功执行的结果
    """
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except:
        return None

def get_git_info():
    """
    获取当前Git信息
    
    返回：
        dict: 包含commit、branch、dirty、message的字典
    
    收集信息：
        - commit: 短commit哈希（7位）
        - branch: 当前分支名
        - dirty: 是否有未提交的修改
        - message: commit消息（首行，截断到80字符）
    """
    info = {}
    info['commit'] = run_command("git rev-parse --short HEAD") or "unknown"
    info['branch'] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

    # 检查仓库是否dirty（有未提交的修改）
    status = run_command("git status --porcelain")
    info['dirty'] = bool(status) if status is not None else False

    # 获取commit消息
    info['message'] = run_command("git log -1 --pretty=%B") or ""
    info['message'] = info['message'].split('\n')[0][:80]  # 首行，截断

    return info

def get_gpu_info():
    """
    获取GPU信息
    
    返回：
        dict: GPU信息字典
        - available: 是否有GPU可用
        - count: GPU数量
        - names: GPU名称列表
        - memory_gb: GPU显存（GB）列表
        - cuda_version: CUDA版本
    
    特性：
        - 自动检测CUDA可用性
        - 获取所有GPU的详细信息
        - 计算总显存
    """
    if not torch.cuda.is_available():
        return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {
        "available": True,
        "count": num_devices,
        "names": [],
        "memory_gb": []
    }

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / (1024**3))

    # 获取CUDA版本
    info["cuda_version"] = torch.version.cuda or "unknown"

    return info

def get_system_info():
    """
    获取系统信息
    
    返回：
        dict: 系统信息字典
    
    收集信息：
        基础系统信息：
            - hostname: 主机名
            - platform: 操作系统（Linux/Windows/macOS）
            - python_version: Python版本
            - torch_version: PyTorch版本
        
        CPU和内存：
            - cpu_count: 物理核心数
            - cpu_count_logical: 逻辑核心数
            - memory_gb: 总内存（GB）
        
        用户和环境：
            - user: 当前用户
            - nanochat_base_dir: nanochat基础目录
            - working_dir: 当前工作目录
    """
    info = {}

    # 基础系统信息
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__

    # CPU和内存
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)

    # 用户和环境
    info['user'] = os.environ.get('USER', 'unknown')
    info['nanochat_base_dir'] = os.environ.get('NANOCHAT_BASE_DIR', 'out')
    info['working_dir'] = os.getcwd()

    return info

def estimate_cost(gpu_info, runtime_hours=None):
    """
    估算训练成本
    
    参数：
        gpu_info: GPU信息字典（从get_gpu_info获取）
        runtime_hours: 运行时长（小时），None=不计算总成本
    
    返回：
        dict或None: 成本估算字典
        - hourly_rate: 每小时成本（美元）
        - gpu_type: GPU类型
        - estimated_total: 估算总成本（如果提供runtime_hours）
    
    定价参考：
        基于Lambda Cloud的粗略定价：
        - H100: $3.00/小时
        - A100: $1.79/小时
        - V100: $0.55/小时
        - 默认: $2.00/小时
    
    注意：
        - 价格会随时间变化
        - 仅作粗略估算参考
    """
    # 粗略定价（来自Lambda Cloud）
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.get("available"):
        return None

    # 尝试从名称识别GPU类型
    hourly_rate = None
    gpu_name = gpu_info["names"][0] if gpu_info["names"] else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * gpu_info["count"]
            break

    if hourly_rate is None:
        hourly_rate = default_rate * gpu_info["count"]  # 默认估算

    return {
        "hourly_rate": hourly_rate,
        "gpu_type": gpu_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None
    }

def generate_header():
    """
    生成训练报告的头部
    
    返回：
        str: Markdown格式的报告头部
    
    包含内容：
        - 生成时间戳
        - Git信息（分支、commit、dirty状态）
        - 硬件信息（CPU、内存、GPU）
        - 软件信息（Python、PyTorch）
        - Bloat指标（代码库统计）
    
    Bloat指标：
        使用files-to-prompt工具打包所有源代码，统计：
        - 字符数、行数、文件数
        - 估算token数（字符数/4）
        - 依赖数量（uv.lock行数）
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    cost_info = estimate_cost(gpu_info)

    header = f"""# nanochat training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info['branch']}
- Commit: {git_info['commit']} {"(dirty)" if git_info['dirty'] else "(clean)"}
- Message: {git_info['message']}

### Hardware
- Platform: {sys_info['platform']}
- CPUs: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)
- Memory: {sys_info['memory_gb']:.1f} GB
"""

    if gpu_info.get("available"):
        gpu_names = ", ".join(set(gpu_info["names"]))
        total_vram = sum(gpu_info["memory_gb"])
        header += f"""- GPUs: {gpu_info['count']}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info['cuda_version']}
"""
    else:
        header += "- GPUs: None available\n"

    if cost_info and cost_info["hourly_rate"] > 0:
        header += f"""- Hourly Rate: ${cost_info['hourly_rate']:.2f}/hour\n"""

    header += f"""
### Software
- Python: {sys_info['python_version']}
- PyTorch: {sys_info['torch_version']}

"""

    # bloat指标：打包所有源代码并评估其大小
    packaged = run_command('files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml')
    num_chars = len(packaged)
    num_lines = len(packaged.split('\n'))
    num_files = len([x for x in packaged.split('\n') if x.startswith('<source>')])
    num_tokens = num_chars // 4  # 假设每token约4个字符

    # 通过uv.lock统计依赖数量
    uv_lock_lines = 0
    if os.path.exists('uv.lock'):
        with open('uv.lock', 'r', encoding='utf-8') as f:
            uv_lock_lines = len(f.readlines())

    header += f"""
### Bloat
- Characters: {num_chars:,}
- Lines: {num_lines:,}
- Files: {num_files:,}
- Tokens (approx): {num_tokens:,}
- Dependencies (uv.lock lines): {uv_lock_lines:,}

"""
    return header

# -----------------------------------------------------------------------------

def slugify(text):
    """
    将文本转换为URL友好的slug格式
    
    参数：
        text: 输入文本
    
    返回：
        slug字符串（小写，空格替换为连字符）
    """
    return text.lower().replace(" ", "-")

# 预期的报告文件及其顺序（完整训练流程）
EXPECTED_FILES = [
    "tokenizer-training.md",       # 分词器训练
    "tokenizer-evaluation.md",     # 分词器评估
    "base-model-training.md",      # 基础模型训练
    "base-model-loss.md",          # 基础模型损失评估
    "base-model-evaluation.md",    # 基础模型评估（CORE）
    "midtraining.md",              # 中期训练
    "chat-evaluation-mid.md",      # 中期模型评估
    "chat-sft.md",                 # 监督微调
    "chat-evaluation-sft.md",      # SFT模型评估
    "chat-rl.md",                  # 强化学习
    "chat-evaluation-rl.md",       # RL模型评估
]

# 我们当前关注的指标
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]

def extract(section, keys):
    """
    从报告section中提取指标值
    
    参数：
        section: 报告section文本
        keys: 要提取的键（字符串或列表）
    
    返回：
        dict: {key: value}字典
    
    提取逻辑：
        查找包含key的行，分割":"获取值
    """
    if not isinstance(keys, list):
        keys = [keys]  # 方便使用
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out

def extract_timestamp(content, prefix):
    """
    从内容中提取时间戳
    
    参数：
        content: 内容文本
        prefix: 时间戳行的前缀（如"timestamp:"）
    
    返回：
        datetime对象，如果解析失败则返回None
    """
    for line in content.split('\n'):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except:
                pass
    return None

class Report:
    """
    训练报告管理器
    
    功能：
        - 维护多个日志section
        - 生成最终的Markdown报告
        - 提取和汇总关键指标
    
    使用流程：
        1. 创建Report对象
        2. 训练过程中调用log()记录各阶段数据
        3. 训练结束后调用generate()生成最终报告
        4. 或调用reset()清除报告开始新实验
    
    属性：
        report_dir: 报告目录路径
    """

    def __init__(self, report_dir):
        """
        初始化报告管理器
        
        参数：
            report_dir: 报告目录路径（自动创建）
        """
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """
        记录一个section的数据
        
        参数：
            section: section名称（如"Base Model Training"）
            data: 数据列表，可以是：
                - 字符串：直接写入
                - 字典：格式化为"- key: value"
                - None/空：跳过
        
        返回：
            file_path: 写入的文件路径
        
        文件格式：
            ## {section}
            timestamp: {当前时间}
            
            - key1: value1
            - key2: value2
        """
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    # 跳过虚假值，如None或空字典等
                    continue
                if isinstance(item, str):
                    # 直接写入字符串
                    f.write(item)
                else:
                    # 渲染字典
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """
        生成最终报告
        
        返回：
            report_file: 生成的报告文件路径
        
        生成流程：
            1. 写入header（系统信息）
            2. 遍历所有预期的section文件
            3. 提取关键指标（base、mid、sft、rl）
            4. 生成Summary表格
            5. 计算总训练时间
            6. 复制到当前目录
        
        输出：
            - {report_dir}/report.md: 完整报告
            - ./report.md: 当前目录副本（方便查看）
        
        特性：
            - 自动提取各阶段指标
            - 生成对比表格
            - 计算墙钟时间
            - 包含Bloat统计
        """
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"正在生成报告到 {report_file}")
        final_metrics = {}  # 最重要的最终指标，将作为表格添加到末尾
        start_time = None
        end_time = None
        with open(report_file, "w", encoding="utf-8") as out_file:
            # 首先写入header
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r", encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
                    # 捕获bloat数据以便后续汇总（Bloat header之后到\n\n之间的内容）
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = None  # 将导致不写入总墙钟时间
                bloat_data = "[bloat数据缺失]"
                print(f"警告: {header_file} 不存在。是否忘记运行 `nanochat reset`?")
            
            # 处理所有单独的section
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"警告: {section_file} 不存在，跳过")
                    continue
                with open(section_file, "r", encoding="utf-8") as in_file:
                    section = in_file.read()
                
                # 从此section提取时间戳（最后一个section的时间戳将"保留"为end_time）
                if "rl" not in file_name:
                    # 跳过RL sections计算end_time，因为RL是实验性的
                    end_time = extract_timestamp(section, "timestamp:")
                
                # 从sections提取最重要的指标
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-mid.md":
                    final_metrics["mid"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(section, "GSM8K")  # RL只评估GSM8K
                
                # 追加此section到报告
                out_file.write(section)
                out_file.write("\n")
            # 添加最终指标表格
            out_file.write("## 总结\n\n")
            
            # 从header复制bloat指标
            out_file.write(bloat_data)
            out_file.write("\n\n")
            
            # 收集所有唯一的指标名称
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            
            # 自定义排序：CORE在前，ChatCORE在后，其他在中间
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
            
            # 固定列宽
            stages = ["base", "mid", "sft", "rl"]
            metric_width = 15
            value_width = 8
            
            # 写入表格头
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            
            # 写入分隔符
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            
            # 写入表格行
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            
            # 计算并写入总墙钟时间
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"总墙钟时间: {hours}小时{minutes}分钟\n")
            else:
                out_file.write("总墙钟时间: 未知\n")
        
        # 同时复制report.md文件到当前目录（方便查看）
        print(f"正在复制report.md到当前目录以便查看")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        """
        重置报告
        
        功能：
            1. 删除所有section文件
            2. 删除report.md（如果存在）
            3. 生成并写入新的header（带开始时间戳）
        
        用途：
            开始新的训练实验前调用，清除旧报告
        
        输出：
            - 删除旧的section文件
            - 生成新的header.md
            - 打印确认信息
        """
        # 删除section文件
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # 删除report.md（如果存在）
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        
        # 生成并写入header section（带开始时间戳）
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"运行开始: {start_time}\n\n---\n\n")
        print(f"已重置报告并将header写入 {header_file}")

# -----------------------------------------------------------------------------
# nanochat专用便捷函数

class DummyReport:
    """
    虚拟报告类（用于非rank 0进程）
    
    功能：
        在分布式训练中，只有rank 0记录报告
        其他rank使用此虚拟类，所有方法都是空操作
    
    方法：
        - log: 空操作
        - reset: 空操作
    
    用途：
        避免多进程重复写入报告，节省开销
    """
    def log(self, *args, **kwargs):
        """空操作：不记录任何内容"""
        pass
    
    def reset(self, *args, **kwargs):
        """空操作：不重置任何内容"""
        pass

def get_report():
    """
    获取报告对象（便捷函数）
    
    返回：
        - 如果是rank 0：返回Report实例（实际记录）
        - 否则：返回DummyReport实例（空操作）
    
    分布式策略：
        仅rank 0进程记录报告，其他进程使用虚拟报告
        避免多进程写入冲突和重复开销
    
    用途：
        在训练脚本中快速获取合适的报告对象
    """
    from nanochat.common import get_base_dir, get_dist_info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        return DummyReport()

if __name__ == "__main__":
    """
    命令行工具主程序
    
    用法：
        python -m nanochat.report generate  # 生成报告
        python -m nanochat.report reset     # 重置报告
        python -m nanochat.report           # 默认：生成报告
    
    命令：
        - generate: 汇总所有section，生成最终报告
        - reset: 清除旧报告，准备新实验
    """
    import argparse
    parser = argparse.ArgumentParser(description="生成或重置nanochat训练报告")
    parser.add_argument("command", nargs="?", default="generate", 
                       choices=["generate", "reset"], 
                       help="要执行的操作（默认：generate）")
    args = parser.parse_args()
    
    if args.command == "generate":
        get_report().generate()
    elif args.command == "reset":
        get_report().reset()
