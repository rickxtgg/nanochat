"""
HumanEval数据集（Python代码生成基准）

HumanEval是OpenAI创建的Python编程问题数据集，用于评估模型的代码生成能力。
数据集名称是误导性的——它与人类无关，而是一个代码生成基准测试。

数据集特点：
    - 164个手写的Python编程问题
    - 每个问题包含函数签名、文档字符串和单元测试
    - 测试模型的代码生成和问题解决能力
    - 评估标准：生成的代码是否通过所有单元测试

数据来源：
    https://huggingface.co/datasets/openai/openai_humaneval

评估方式：
    - 在沙箱环境中执行生成的代码
    - 运行预定义的单元测试
    - 通过所有测试则认为成功

参考文献：
    Evaluating Large Language Models Trained on Code
    https://arxiv.org/abs/2107.03374

注意：
    数据集名称"HumanEval"具有误导性，与人类评估无关，
    实际上是自动化的代码执行和测试评估。
"""

import re  # 正则表达式（用于提取代码块）
from datasets import load_dataset  # HuggingFace datasets库
from nanochat.execution import execute_code  # 沙箱代码执行
from tasks.common import Task  # 任务基类

def extract_imports(prompt):
    """
    从代码块的开头提取import语句
    
    参数：
        prompt: 代码文本
    
    返回：
        所有import语句（用换行符连接）
    
    逻辑：
        - 从头开始遍历每一行
        - 收集以"import "或"from "开头的行
        - 遇到第一个非import、非注释行时停止
    
    示例：
        >>> extract_imports("import math\nfrom typing import List\n\ndef foo():\n    pass")
        'import math\nfrom typing import List'
    """
    imports = []
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped and not stripped.startswith('#'):
            # 遇到第一个非import、非注释行时停止
            break
    return '\n'.join(imports)

def extract_program(completion):
    """
    从LLM生成的文本中提取Python代码
    
    参数：
        completion: LLM生成的完整文本
    
    返回：
        提取的Python代码
    
    处理各种输出格式：
        - 用```python ... ```或``` ... ```包装的代码块
        - 没有markdown块的纯代码
        - 代码块前后有额外文本
    
    逻辑：
        1. 首先尝试查找markdown代码块
        2. 如果找到，返回第一个代码块
        3. 如果未找到，返回整个completion
    
    示例：
        >>> extract_program("```python\ndef foo():\n    pass\n```")
        'def foo():\n    pass'
        >>> extract_program("def foo():\n    pass")
        'def foo():\n    pass'
    """
    # 尝试查找markdown代码块（```python或只是```）
    # 匹配```python\n...\n```或```\n...\n```
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)

    if matches:
        # 返回找到的第一个代码块
        return matches[0].strip()

    # 未找到代码块，返回整个completion
    return completion.strip()

class HumanEval(Task):
    """
    HumanEval数据集任务类
    
    特性：
        - 评估类型：generative（生成式）
        - 164个Python编程问题
        - 只有test分割（没有训练集）
        - 自动化评估：在沙箱中执行代码并运行测试
    
    数据结构：
        - prompt: 函数签名和文档字符串
        - canonical_solution: 标准解答
        - entry_point: 要测试的函数名
        - test: 单元测试代码
    
    用法示例：
        # 加载HumanEval测试集
        task = HumanEval()
        
        # 加载前10个样本
        task = HumanEval(start=0, stop=10)
    
    安全性：
        评估时使用沙箱环境（nanochat.execution）执行代码，
        确保不会执行危险操作。
    """

    def __init__(self, **kwargs):
        """
        初始化HumanEval任务
        
        参数：
            **kwargs: 传递给父类Task的参数（start、stop、step）
        
        注意：
            HumanEval只有test分割，没有train分割
        """
        super().__init__(**kwargs)
        # 加载数据集并打乱（使用固定种子以保证可复现性）
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        """
        评估类型
        
        返回：
            'generative': 表示这是生成式任务（需要生成完整代码）
        """
        return 'generative'

    def num_examples(self):
        """
        获取数据集中的样本总数
        
        返回：
            样本数量（整数，HumanEval有164个问题）
        """
        return len(self.ds)

    def get_example(self, index):
        """
        获取指定索引的样本
        
        参数：
            index: 样本索引
        
        返回：
            conversation: 对话字典，包含：
                - messages: 消息列表（user提示 + assistant解答）
                - entry_point: 函数入口点（评估时需要）
                - test: 测试用例（评估时需要）
        
        数据格式：
            - prompt: HumanEval中的提示是程序的开头部分
            - solution: 程序的正确续写
            - complete_solution: prompt + solution的完整解答
        """
        row = self.ds[index]
        prompt = row['prompt']  # HumanEval中的提示是程序的开头
        solution = row['canonical_solution']  # 程序的正确续写
        entry_point = row['entry_point']  # 要检查的函数名
        test = row['test']  # 测试用例
        complete_solution = f"{prompt}\n{solution}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]
        conversation = {
            "messages": messages,
            "entry_point": entry_point,  # 评估时需要
            "test": test,  # 评估时需要
        }
        return conversation

    def evaluate(self, conversation, completion):
        """
        评估生成的代码是否正确
        
        参数：
            conversation: 对话字典（包含测试用例）
            completion: 生成的代码
        
        返回：
            bool: True表示通过所有测试，False表示失败
        
        评估流程：
            1. 从prompt中提取import语句
            2. 从completion中提取代码
            3. 组合：imports + 生成的代码 + 测试用例 + check调用
            4. 在沙箱中执行完整程序
            5. 返回执行成功与否
        
        注意：
            - prompt包含imports和函数签名
            - completion通常包含整个函数，但不一定有imports
            - 因此我们手动添加imports
        """
        # prompt包含imports和函数签名
        imports = extract_imports(conversation['messages'][0]['content'])
        # completion通常包含整个函数，但不一定有所需的imports，所以我们手动添加
        completion_code = extract_program(completion)
        # 组合完整程序
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation['test']
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        # 在沙箱中执行代码
        result = execute_code(program)
        success = result.success
        return success
