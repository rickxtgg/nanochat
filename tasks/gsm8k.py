"""
GSM8K数据集（小学数学应用题）

GSM8K（Grade School Math 8K）是一个包含8500个高质量小学数学应用题的数据集，
由OpenAI创建，用于测试模型的数学推理和计算能力。

数据集特点：
    - 8,500个小学数学应用题
    - 需要2-8步的多步推理
    - 使用自然语言描述解题过程
    - 支持工具调用（计算器）
    - 答案标记在#### 之后

数据来源：
    https://huggingface.co/datasets/openai/gsm8k

示例问题：
    问题：
    Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes 
    of babysitting. How much did she earn?
    
    答案：
    Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
    Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
    #### 10

工具调用格式：
    注意GSM8K在<< >>标签内使用工具调用（计算器）。
    格式：<<表达式=结果>>
    例如：<<12/60=0.2>> 表示计算12/60，结果是0.2

参考文献：
    Training Verifiers to Solve Math Word Problems
    https://arxiv.org/abs/2110.14168
"""

import re  # 正则表达式（用于提取答案）
from datasets import load_dataset  # HuggingFace datasets库
from tasks.common import Task  # 任务基类


# 正则表达式：匹配#### 后的数值答案
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    """
    提取#### 标记后的数值答案
    
    参数：
        completion: 完整的回答文本
    
    返回：
        提取的数值答案（字符串），如果未找到则返回None
    
    标准化处理：
        - 移除逗号分隔符（如1,000 -> 1000）
        - 保留负号和小数点
    
    参考：
        遵循官方代码的标准化处理
        https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    
    示例：
        >>> extract_answer("...some text...#### 10")
        '10'
        >>> extract_answer("...some text...#### 1,234.56")
        '1234.56'
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")  # 移除千位分隔符
        return match_str
    return None


class GSM8K(Task):
    """
    GSM8K数据集任务类
    
    特性：
        - 评估类型：generative（生成式）
        - 支持两个子集：main（主数据集）和socratic（苏格拉底式对话）
        - 支持两个数据分割：train（训练集7.5K）、test（测试集1K）
        - 支持工具调用：自动解析<< >>标签内的计算器调用
    
    工具调用处理：
        GSM8K的答案中包含计算器调用，格式为<<表达式=结果>>
        本类会自动将这些调用解析为结构化的parts：
        - {"type": "python", "text": "表达式"}
        - {"type": "python_output", "text": "结果"}
    
    用法示例：
        # 加载GSM8K主数据集训练集
        task = GSM8K(subset="main", split="train")
        
        # 加载测试集的前100个样本
        task = GSM8K(subset="main", split="test", start=0, stop=100)
    """

    def __init__(self, subset, split, **kwargs):
        """
        初始化GSM8K任务
        
        参数：
            subset: 子集名称，必须是"main"或"socratic"
            split: 数据分割，必须是"train"或"test"
            **kwargs: 传递给父类Task的参数（start、stop、step）
        """
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K子集必须是main或socratic"
        assert split in ["train", "test"], "GSM8K分割必须是train或test"
        # 加载数据集并打乱（使用固定种子以保证可复现性）
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        """
        评估类型
        
        返回：
            'generative': 表示这是生成式任务（需要生成完整解答）
        """
        return 'generative'

    def num_examples(self):
        """
        获取数据集中的样本总数
        
        返回：
            样本数量（整数）
        """
        return len(self.ds)

    def get_example(self, index):
        """
        获取指定索引的样本
        
        参数：
            index: 样本索引
        
        返回：
            conversation: 对话字典，包含：
                - messages: 消息列表（user问题 + assistant解答）
        
        工具调用解析：
            这里比较复杂，因为GSM8K使用工具调用，我们需要在这里解析。
            将答案中的<<表达式=结果>>标签解析为：
            1. {"type": "python", "text": "表达式"}
            2. {"type": "python_output", "text": "结果"}
        
        数据格式：
            user消息：简单字符串（问题）
            assistant消息：parts列表（文本+工具调用混合）
        """
        row = self.ds[index]
        question = row['question']  # 问题文本（字符串）
        answer = row['answer']  # 完整解答和#### 标记后的答案（字符串）
        
        # 创建并返回对话对象
        # 这里比较复杂，因为GSM8K使用工具调用，我们需要在这里解析
        assistant_message_parts = []
        # 使用正则表达式分割文本和工具调用
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # 这是一个计算器工具调用
                inner = part[2:-2]  # 移除<< >>
                # 按=分割以获取表达式和结果
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # 添加工具调用作为一个part
                assistant_message_parts.append({"type": "python", "text": expr})
                # 添加结果作为一个part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # 工具调用之间的常规文本
                assistant_message_parts.append({"type": "text", "text": part})
        
        # 现在将所有内容组合在一起
        messages = [
            {"role": "user", "content": question},  # 注意：简单字符串
            {"role": "assistant", "content": assistant_message_parts},  # 注意：parts列表（字典）
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        评估assistant的回答是否正确
        
        参数：
            conversation: 对话字典（包含正确答案）
            assistant_response: assistant的回答（通常是通过采样获得的替代回答）
        
        返回：
            int: 0表示错误，1表示正确
        
        注意：
            - conversation包含user消息和assistant消息（包含ground truth答案）
            - assistant_response通常是通过采样获得的替代assistant消息
        
        TODO：
            技术上，assistant_response应该是一个Message（字符串或parts列表）
            我们以后可能会处理这个。现在假设是字符串。
        
        评估逻辑：
            1. 从ground truth中提取答案（最后一个text part中的#### 后的数值）
            2. 从预测回答中提取答案
            3. 比较两个数值是否相等
        """
        assert isinstance(assistant_response, str), "目前假设是简单字符串回答"
        
        # 首先提取ground truth答案
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "最后一条消息必须来自Assistant"
        assert isinstance(assistant_message['content'], list), "期望这是一个parts列表"
        last_text_part = assistant_message['content'][-1]['text']  # 这包含GSM8K中的最终答案
        
        # 提取ground truth答案和预测答案
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        
        # 比较并返回成功结果（整数）
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        用于强化学习的奖励函数
        
        参数：
            conversation: 对话字典
            assistant_response: assistant的回答
        
        返回：
            float: 奖励值（1.0=正确，0.0=错误）
        
        实现：
            为了保持简单，只是重用上面的评估函数。
            以后可以使其更复杂（例如格式匹配等）
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float
