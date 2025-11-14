"""
SmolTalk数据集（通用对话数据集）

SmolTalk是HuggingFace创建的高质量对话数据集，特别适合训练较小的对话模型。
这是一个"通用"对话数据集，涵盖各种日常对话场景。

数据集特点：
    - 训练集：约460K对话
    - 测试集：约24K对话
    - 高质量的多轮对话
    - 支持可选的system消息
    - 角色交替（user/assistant）
    - 适合较小模型训练

数据来源：
    https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk

版本选择：
    我们使用"smol"版本，它更适合较小的模型，
    相比完整版本有更好的质量/数量平衡。

用途：
    - SFT训练的主要对话数据
    - 提高模型的对话能力
    - 学习多轮对话交互模式
"""

from datasets import load_dataset  # HuggingFace datasets库
from tasks.common import Task  # 任务基类

class SmolTalk(Task):
    """
    SmolTalk数据集任务类
    
    数据规模：
        - 训练集：460K对话
        - 测试集：24K对话
    
    特性：
        - 多轮对话（至少2轮）
        - 可选的system消息
        - 严格的角色交替（user/assistant）
        - 高质量的自然对话
    
    用法示例：
        # 加载训练集
        task = SmolTalk(split="train")
        
        # 加载测试集的前1000个样本
        task = SmolTalk(split="test", start=0, stop=1000)
    
    注意：
        此数据集不需要评估函数，主要用于SFT训练
    """

    def __init__(self, split, **kwargs):
        """
        初始化SmolTalk任务
        
        参数：
            split: 数据分割，必须是"train"或"test"
            **kwargs: 传递给父类Task的参数（start、stop、step）
        """
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk分割必须是train或test"
        # 加载数据集并打乱（使用固定种子以保证可复现性）
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        """
        获取数据集中的对话总数
        
        返回：
            对话数量（整数）
        """
        return self.length

    def get_example(self, index):
        """
        获取指定索引的对话
        
        参数：
            index: 对话索引
        
        返回：
            conversation: 对话字典，包含：
                - messages: 消息列表
        
        数据验证：
            - 至少1条消息（如果有system消息）
            - 去除system消息后至少2条消息
            - 角色必须交替（user/assistant/user/assistant...）
            - 消息内容必须是字符串
        
        消息格式：
            可选的system消息：
            [{"role": "system", "content": "..."}, ...]
            
            必需的user/assistant交替：
            [{"role": "user", "content": "..."},
             {"role": "assistant", "content": "..."},
             ...]
        """
        row = self.ds[index]
        messages = row["messages"]
        
        # ---------------------------------------------------------------------
        # 这里进行合理性检查
        # TODO: 以后可以移除这些assert，现在只是不想有任何错误
        
        # 开头可以有一个可选的system消息
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]  # 可选的system消息是可以的
        else:
            rest_messages = messages
        
        # 必须至少有2条消息（user和assistant各一条）
        assert len(rest_messages) >= 2, "SmolTalk消息必须至少有2条"
        
        # 验证消息结构和角色交替
        for i, message in enumerate(rest_messages):
            # user和assistant交替，如user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, \
                f"消息{i}的角色是{message['role']}，但应该是{expected_role}"
            assert isinstance(message["content"], str), "内容必须是字符串"
        
        # ---------------------------------------------------------------------
        # 创建并返回对话对象（可以包含system消息）
        conversation = {
            "messages": messages,
        }
        return conversation
