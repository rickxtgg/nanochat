"""
ARC数据集（Allen AI推理挑战赛）

这是来自Allen AI的ARC（AI2 Reasoning Challenge）数据集。
ARC是一个多项选择题科学问答数据集，旨在测试模型的推理能力。

数据集特点：
    - 包含两个子集：ARC-Easy和ARC-Challenge
    - 问题来自3-9年级的科学考试
    - ARC-Challenge包含更难的问题，简单的信息检索方法无法解决
    - 每个问题有3-5个选项

数据来源：
    https://huggingface.co/datasets/allenai/ai2_arc

参考文献：
    Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
    https://arxiv.org/abs/1803.05457
"""

from datasets import load_dataset  # HuggingFace datasets库
from tasks.common import Task, render_mc  # 任务基类和多选题渲染函数

class ARC(Task):
    """
    ARC数据集任务类
    
    特性：
        - 评估类型：categorical（分类/多选题）
        - 支持两个难度子集：ARC-Easy和ARC-Challenge
        - 支持三个数据分割：train、validation、test
        - 自动打乱数据（seed=42，可复现）
    
    用法示例：
        # 加载ARC-Easy训练集
        task = ARC(subset="ARC-Easy", split="train")
        
        # 加载ARC-Challenge验证集的前100个样本
        task = ARC(subset="ARC-Challenge", split="validation", start=0, stop=100)
    """

    def __init__(self, subset, split, **kwargs):
        """
        初始化ARC任务
        
        参数：
            subset: 子集名称，必须是"ARC-Easy"或"ARC-Challenge"
            split: 数据分割，必须是"train"、"validation"或"test"
            **kwargs: 传递给父类Task的参数（start、stop、step）
        """
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC子集必须是ARC-Easy或ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC分割必须是train、validation或test"
        # 加载数据集并打乱（使用固定种子以保证可复现性）
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        """
        评估类型
        
        返回：
            'categorical': 表示这是分类任务（多选题）
        """
        return 'categorical'

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
                - messages: 消息列表（user问题 + assistant答案）
                - letters: 选项字母列表（用于评估时限制预测范围）
        
        数据格式：
            问题：Multiple Choice question: {question}
            选项：- {choice}={letter}
            答案：单个字母（如"A"）
        """
        row = self.ds[index]
        question = row["question"]  # 问题文本
        choices = row["choices"]["text"]  # 每个选项的文本
        answer_string = row["answerKey"]  # 正确答案的字母，如"A"、"B"、"C"、"D"
        letters = row["choices"]["label"]  # 选项字母列表，如["A", "B", "C", "D"]
        
        # 合理性检查：确保答案在选项中
        assert answer_string in letters, f"ARC答案{answer_string}必须是{letters}之一"
        
        # 创建并返回对话对象
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            # 在评估时有用，可以将assistant的预测限制到这些字母之一
            "letters": letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        评估assistant的回答是否正确
        
        参数：
            conversation: 对话字典（包含正确答案）
            assistant_response: assistant的回答（应该是单个字母）
        
        返回：
            bool: True表示回答正确，False表示错误
        
        注意：
            这里的assert严格来说不是必需的，但根据当前的评估方式，
            我们期望这个条件为真。保留assert以防止错误，
            但将来可能会移除它。
        """
        assert assistant_response in conversation['letters'], \
            f"ARC答案{assistant_response}应该是{conversation['letters']}之一"
        assistant_message = conversation['messages'][-1]['content']  # 正确答案，如"A"
        return assistant_response == assistant_message
