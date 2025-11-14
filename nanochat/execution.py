"""
沙箱化Python代码执行工具

用于安全地运行LLM生成的Python代码。
改编自OpenAI HumanEval项目：
https://github.com/openai/human-eval/blob/master/human_eval/execution.py

已实现的安全措施：
    1. 进程隔离：
       - 每次执行在独立进程中运行
       - 进程可以被强制终止（如果挂起或崩溃）
    
    2. 超时限制：
       - 默认5秒超时
       - 防止无限循环
       - 使用signal.SIGALRM实现
    
    3. 内存限制：
       - 默认256MB内存限制
       - 使用resource模块强制执行（Linux）
       - macOS (Darwin)系统跳过内存限制
    
    4. I/O捕获：
       - 捕获stdout和stderr
       - 禁用stdin（防止阻塞）
       - 所有输出返回给调用者
    
    5. 临时目录：
       - 代码在临时目录中运行
       - 执行后自动删除
       - 避免污染文件系统
    
    6. 危险函数禁用：
       - 禁用破坏性系统调用（os.system、os.kill、subprocess.Popen等）
       - 禁用文件操作（os.remove、shutil.rmtree等）
       - 禁用内置函数（exit、quit、help等）

未实现的安全措施（局限性）：
    ⚠️ 这不是一个真正的安全沙箱！
    
    1. 网络访问未阻止：
       - socket连接仍然可以建立
       - 可以发起HTTP/HTTPS请求
    
    2. Python动态特性可绕过限制：
       - ctypes可以调用系统库
       - __builtins__可以被恢复
       - eval/exec可以执行任意代码
    
    3. 无内核级隔离：
       - 没有seccomp过滤
       - 没有容器隔离
       - 没有虚拟化

使用场景：
    ✅ 适用：评估生成代码、防止意外破坏性行为
    ❌ 不适用：对抗恶意对抗性代码

典型用例：
    - HumanEval代码生成评估
    - 算法竞赛题目测试
    - 教育场景的代码执行
    - LLM工具使用能力评估
"""

import contextlib  # 上下文管理器工具
import faulthandler  # 错误处理器（用于禁用）
import io  # I/O流操作
import multiprocessing  # 多进程（进程隔离）
import os  # 操作系统接口
import platform  # 平台信息（检测macOS）
import signal  # Unix信号（超时控制）
import tempfile  # 临时文件/目录
from dataclasses import dataclass  # 数据类装饰器
from typing import Optional  # 类型提示

# =============================================================================
# 数据结构和上下文管理器
# =============================================================================

@dataclass
class ExecutionResult:
    """
    代码执行结果
    
    字段：
        success: 是否成功执行（无异常、无超时、无内存错误）
        stdout: 标准输出内容
        stderr: 标准错误内容
        error: 错误消息（如果有）
        timeout: 是否超时
        memory_exceeded: 是否超出内存限制
    
    用途：
        封装所有执行信息，便于调用者判断执行状态
        支持友好的字符串表示（__repr__）
    """
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False

    def __repr__(self):
        """友好的字符串表示，只显示非默认值字段"""
        parts = []
        parts.append(f"ExecutionResult(success={self.success}")
        if self.timeout:
            parts.append(", timeout=True")
        if self.memory_exceeded:
            parts.append(", memory_exceeded=True")
        if self.error:
            parts.append(f", error={self.error!r}")
        if self.stdout:
            parts.append(f", stdout={self.stdout!r}")
        if self.stderr:
            parts.append(f", stderr={self.stderr!r}")
        parts.append(")")
        return "".join(parts)


@contextlib.contextmanager
def time_limit(seconds: float):
    """
    超时上下文管理器
    
    参数：
        seconds: 超时时长（秒，支持浮点数）
    
    功能：
        使用Unix信号SIGALRM在指定时间后触发TimeoutException
        适用于防止代码无限循环或长时间阻塞
    
    实现：
        - setitimer: 设置定时器（支持浮点数，精度更高）
        - SIGALRM: 定时器到期时发送的信号
        - 退出时清除定时器（设置为0）
    
    注意：
        - 仅适用于Unix系统（Linux、macOS）
        - Windows不支持signal.SIGALRM
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)  # 设置实时定时器
    signal.signal(signal.SIGALRM, signal_handler)  # 注册信号处理器
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)  # 清除定时器


@contextlib.contextmanager
def capture_io():
    """
    捕获I/O流上下文管理器
    
    功能：
        - 捕获stdout（标准输出）
        - 捕获stderr（标准错误）
        - 禁用stdin（标准输入，防止阻塞）
    
    返回：
        (stdout_capture, stderr_capture): 两个StringIO对象
    
    用途：
        获取代码执行的所有输出，而不是打印到终端
        防止代码尝试读取输入而阻塞
    """
    stdout_capture = io.StringIO()  # 捕获stdout
    stderr_capture = io.StringIO()  # 捕获stderr
    stdin_block = WriteOnlyStringIO()  # 阻止stdin读取
    with contextlib.redirect_stdout(stdout_capture):
        with contextlib.redirect_stderr(stderr_capture):
            with redirect_stdin(stdin_block):
                yield stdout_capture, stderr_capture


@contextlib.contextmanager
def create_tempdir():
    """
    创建临时目录并切换到该目录
    
    功能：
        1. 创建临时目录
        2. 切换工作目录到临时目录
        3. 执行完成后自动删除临时目录
        4. 恢复原工作目录
    
    用途：
        隔离代码执行环境，防止污染文件系统
        自动清理生成的临时文件
    """
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    """超时异常：代码执行超出时间限制"""
    pass


class WriteOnlyStringIO(io.StringIO):
    """
    只写StringIO（阻止读取）
    
    功能：
        重写所有读取方法，抛出IOError
        用于替换stdin，防止代码尝试读取输入而阻塞
    
    设计原理：
        LLM生成的代码不应该需要用户输入
        如果代码尝试input()或sys.stdin.read()，应该立即失败
    """

    def read(self, *args, **kwargs):
        """阻止读取：抛出IOError"""
        raise IOError

    def readline(self, *args, **kwargs):
        """阻止按行读取：抛出IOError"""
        raise IOError

    def readlines(self, *args, **kwargs):
        """阻止读取所有行：抛出IOError"""
        raise IOError

    def readable(self, *args, **kwargs):
        """标记为不可读"""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    """重定向stdin的上下文管理器（类似redirect_stdout/stderr）"""
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    """
    临时切换工作目录
    
    参数：
        root: 目标目录路径
    
    功能：
        切换到指定目录，执行完成后恢复原目录
        如果root为"."则不切换
    
    实现：
        保存当前目录 → 切换目录 → yield → 恢复目录
    """
    if root == ".":
        yield  # 无需切换
        return
    cwd = os.getcwd()  # 保存当前目录
    os.chdir(root)  # 切换到目标目录
    try:
        yield
    finally:
        os.chdir(cwd)  # 恢复原目录


# =============================================================================
# 可靠性保护和安全措施
# =============================================================================

def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    可靠性保护：禁用破坏性函数
    
    参数：
        maximum_memory_bytes: 最大内存限制（字节），None=不限制
    
    功能：
        禁用各种破坏性函数，防止生成代码干扰测试环境
        包括：
        - 进程管理（fork、kill、killpg等）
        - 文件操作（remove、rmtree、chmod等）
        - 系统调用（system、putenv、chroot等）
        - 内置函数（exit、quit、help等）
        - 危险模块（subprocess、psutil、resource等）
    
    保护措施：
        1. 设置内存限制（resource.RLIMIT_AS/DATA/STACK）
        2. 禁用faulthandler（避免干扰信号处理）
        3. 将危险函数设置为None（调用时会TypeError）
        4. 阻止导入危险模块（设置sys.modules[module]=None）
        5. 设置OMP_NUM_THREADS=1（避免过度并行）
    
    ⚠️ 警告：
        这不是一个安全沙箱！
        不信任的代码（包括模型生成的代码）不应该在没有真正沙箱的情况下盲目执行。
        参见OpenAI Codex论文了解真正的代码沙箱实现。
        使用时请谨慎！
    
    局限性：
        - 可以通过ctypes、__builtins__等绕过
        - 不阻止网络访问
        - 不阻止文件读取
        - macOS上内存限制可能失效
    """

    # 设置内存限制（仅Linux，macOS上似乎会失败）
    if platform.uname().system != "Darwin":
        import resource
        # RLIMIT_AS: 地址空间大小限制
        # RLIMIT_DATA: 数据段大小限制
        # RLIMIT_STACK: 栈大小限制
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    # 禁用faulthandler（避免干扰我们的信号处理）
    faulthandler.disable()

    # 禁用内置函数
    import builtins
    builtins.exit = None  # 禁止退出
    builtins.quit = None  # 禁止退出

    # 禁用os模块的危险函数
    import os
    os.environ["OMP_NUM_THREADS"] = "1"  # 限制OpenMP线程数

    # 进程管理相关（防止fork炸弹、杀死其他进程）
    os.kill = None
    os.system = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.setuid = None
    
    # 文件操作相关（防止删除/修改文件）
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    
    # 权限修改相关
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    
    # 目录操作相关
    os.fchdir = None
    os.chroot = None
    os.fchdir = None  # 重复，但保留原代码结构
    os.getcwd = None
    os.chdir = None

    # 禁用shutil模块的危险函数
    import shutil
    shutil.rmtree = None  # 防止递归删除目录
    shutil.move = None  # 防止移动文件
    shutil.chown = None  # 防止修改所有权

    # 禁用subprocess模块
    import subprocess
    subprocess.Popen = None  # type: ignore  # 防止执行外部命令

    # 禁用help函数
    __builtins__["help"] = None

    # 阻止导入危险模块（设置为None）
    import sys
    sys.modules["ipdb"] = None  # 调试器
    sys.modules["joblib"] = None  # 并行处理
    sys.modules["resource"] = None  # 资源限制（防止修改）
    sys.modules["psutil"] = None  # 进程和系统工具
    sys.modules["tkinter"] = None  # GUI库


# =============================================================================
# 代码执行核心函数
# =============================================================================

def _unsafe_execute(code: str, timeout: float, maximum_memory_bytes: Optional[int], result_dict):
    """
    在子进程中执行代码（带安全保护）
    
    参数：
        code: 要执行的Python代码字符串
        timeout: 超时时长（秒）
        maximum_memory_bytes: 内存限制（字节）
        result_dict: 共享字典（用于将结果传回父进程）
    
    功能：
        1. 创建临时目录并切换到该目录
        2. 保存清理临时目录所需的系统调用（在禁用前）
        3. 应用可靠性保护（禁用危险函数）
        4. 捕获I/O流（stdout/stderr）
        5. 在超时限制内执行代码
        6. 处理各种异常（超时、内存错误、一般异常）
        7. 恢复系统调用以清理临时目录
        8. 将结果写入result_dict
    
    注意：
        此函数在独立进程中运行，无法直接返回值
        必须通过multiprocessing.Manager().dict()共享结果
    """
    with create_tempdir():

        # 保存清理临时目录所需的系统调用
        # （必须在reliability_guard()禁用它们之前保存）
        import os
        import shutil
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink

        # 应用可靠性保护：禁用破坏性函数
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)

        # 默认结果为失败（如果执行成功会更新）
        result_dict.update({
            "success": False,
            "stdout": "",
            "stderr": "",
            "timeout": False,
            "memory_exceeded": False,
            "error": None,
        })

        try:
            exec_globals = {}  # 空的全局命名空间（隔离执行环境）
            with capture_io() as (stdout_capture, stderr_capture):
                with time_limit(timeout):
                    # ⚠️ 警告：
                    # 本程序用于执行不受信任的模型生成代码。
                    # 虽然模型生成代码主动恶意的可能性很小，但由于模型能力或对齐不足，
                    # 生成的代码可能会执行破坏性操作。
                    # 
                    # 强烈建议用户在沙箱环境中运行此评估套件，
                    # 以防止对主机或网络造成破坏性操作。
                    # 
                    # 有关OpenAI如何沙箱化代码的更多信息，请参阅相关论文。
                    # 一旦您阅读了此免责声明并采取了适当的预防措施，
                    # 请取消注释以下行并自行承担风险：
                    exec(code, exec_globals)

            # 执行成功：更新结果
            result_dict.update({
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            })

        except TimeoutException:
            # 超时异常
            result_dict.update({
                "timeout": True,
                "error": "Execution timed out",
            })

        except MemoryError as e:
            # 内存错误
            result_dict.update({
                "memory_exceeded": True,
                "error": f"Memory limit exceeded: {e}",
            })

        except BaseException as e:
            # 其他所有异常（语法错误、运行时错误等）
            result_dict.update({
                "error": f"{type(e).__name__}: {e}",
            })

        # 恢复系统调用（清理临时目录需要）
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink


def execute_code(
    code: str,
    timeout: float = 5.0,  # 默认5秒超时
    maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024,  # 默认256MB内存限制
) -> ExecutionResult:
    """
    在沙箱环境中执行Python代码（主入口函数）
    
    参数：
        code: 要执行的Python代码字符串
        timeout: 最大执行时间（秒，默认5.0）
        maximum_memory_bytes: 内存限制（字节，默认256MB，None=无限制）
    
    返回：
        ExecutionResult对象，包含：
        - success: 是否成功
        - stdout: 标准输出
        - stderr: 标准错误
        - error: 错误消息（如果有）
        - timeout: 是否超时
        - memory_exceeded: 是否超出内存限制
    
    工作流程：
        1. 创建multiprocessing.Manager()管理的共享字典
        2. 在独立进程中运行_unsafe_execute()
        3. 父进程等待子进程完成（最多timeout+1秒）
        4. 如果子进程仍在运行：强制终止并返回超时结果
        5. 如果result_dict为空：返回失败结果
        6. 否则：从result_dict构造ExecutionResult返回
    
    进程隔离优势：
        - 子进程挂起/崩溃不影响父进程
        - 可以强制终止无限循环的代码
        - 内存限制在子进程中生效
        - 子进程退出后自动清理资源
    
    示例：
        >>> result = execute_code("print('hello world')")
        >>> result.success
        True
        >>> result.stdout
        'hello world\\n'
        
        >>> result = execute_code("while True: pass", timeout=1.0)
        >>> result.timeout
        True
    """

    # 创建进程间共享字典
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    # 在独立进程中执行代码
    p = multiprocessing.Process(
        target=_unsafe_execute,
        args=(code, timeout, maximum_memory_bytes, result_dict)
    )
    p.start()  # 启动子进程
    p.join(timeout=timeout + 1)  # 等待子进程（最多timeout+1秒）

    # 检查子进程是否仍在运行（超时）
    if p.is_alive():
        p.kill()  # 强制终止进程
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution timed out (process killed)",
            timeout=True,
            memory_exceeded=False,
        )

    # 检查是否有结果返回（子进程可能崩溃）
    if not result_dict:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution failed (no result returned)",
            timeout=True,
            memory_exceeded=False,
        )

    # 从共享字典构造结果对象
    return ExecutionResult(
        success=result_dict["success"],
        stdout=result_dict["stdout"],
        stderr=result_dict["stderr"],
        error=result_dict["error"],
        timeout=result_dict["timeout"],
        memory_exceeded=result_dict["memory_exceeded"],
    )

