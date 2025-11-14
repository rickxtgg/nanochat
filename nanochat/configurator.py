"""
极简配置器（Poor Man's Configurator）

这是一个非常简单但可能有争议的配置系统。

设计理念：
    作者不喜欢复杂的配置系统，也不喜欢在每个变量前都加config.前缀。
    这个configurator允许直接覆盖全局变量。

使用方式：
    $ python train.py config/override_file.py --batch_size=32
    
    这会：
    1. 首先运行config/override_file.py中的代码
    2. 然后将batch_size覆盖为32

工作原理：
    这个文件会在训练脚本中通过exec()执行：
    >>> exec(open('configurator.py').read())
    
    所以它不是Python模块，而是将配置逻辑从train.py中分离出来。
    这段代码会覆盖globals()中的变量。

注意：
    - 这不是一个标准的Python模块
    - 它通过exec()动态执行
    - 直接修改调用者的全局变量
    - 配置文件（.py）也会被exec()执行

示例：
    # config/my_config.py
    batch_size = 32
    learning_rate = 1e-3
    
    # 运行
    $ python train.py config/my_config.py --learning_rate=1e-4
    # 最终: batch_size=32, learning_rate=1e-4

承认的问题：
    作者知道人们可能不会喜欢这种方式，但如果有人能提出
    更好的简单Python解决方案，作者愿意倾听。
"""

import os
import sys
from ast import literal_eval

# =============================================================================
# 辅助函数
# =============================================================================

def print0(s="", **kwargs):
    """只在rank 0打印（避免分布式训练中的重复输出）"""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

# =============================================================================
# 主配置逻辑：解析命令行参数并覆盖全局变量
# =============================================================================

for arg in sys.argv[1:]:
    if '=' not in arg:
        # ===== 情况1：配置文件 =====
        # 假设这是配置文件的名称（不以--开头）
        assert not arg.startswith('--')
        config_file = arg
        print0(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print0(f.read())
        # 执行配置文件中的Python代码（会修改全局变量）
        exec(open(config_file).read())
    else:
        # ===== 情况2：命令行覆盖 =====
        # 假设这是--key=value格式的参数
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]  # 移除'--'前缀
        
        if key in globals():
            try:
                # 尝试将值评估为Python字面量（如bool、数字等）
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # 如果失败，就使用字符串
                attempt = val
            
            # 确保类型匹配
            if globals()[key] is not None:
                attempt_type = type(attempt)
                default_type = type(globals()[key])
                assert attempt_type == default_type, f"Type mismatch: {attempt_type} != {default_type}"
            
            # 覆盖全局变量
            print0(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
