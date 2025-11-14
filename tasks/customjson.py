"""
CustomJSON任务：从JSONL文件加载对话

这个任务类允许从自定义的JSONL（JSON Lines）文件加载对话数据。
每行应该是一个JSON数组，包含消息对象。

文件格式：
    每行一个JSON数组，代表一个完整的对话
    每个消息对象必须包含'role'和'content'字段
    角色必须交替出现（user -> assistant -> user -> ...）

示例行：
    [{"role":"user","content":"你好"},{"role":"assistant","content":"你好！有什么可以帮你的吗？"}]
    [{"role":"user","content":"今天天气怎么样？"},{"role":"assistant","content":"我无法获取实时天气信息。"}]

用途：
    - 加载自定义对话数据集
    - SFT训练时使用自定义对话
    - 身份对话（identity conversations）
    - 特定领域的对话数据
"""

import os  # 文件操作
import json  # JSON解析
from tasks.common import Task  # 任务基类

class CustomJSON(Task):
    """
    从JSONL文件加载对话的任务类
    
    功能：
        - 从JSONL文件读取对话数据
        - 验证对话结构的正确性
        - 提供统一的Task接口
    
    文件要求：
        1. JSONL格式（每行一个JSON对象）
        2. 每行是一个消息数组
        3. 每个消息必须有'role'和'content'字段
        4. 角色必须交替（user/assistant）
        5. 至少包含2条消息
    
    示例行格式：
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]
    
    用法示例：
        # 加载自定义对话文件
        task = CustomJSON("data/my_conversations.jsonl")
        
        # 加载并切片
        task = CustomJSON("data/my_conversations.jsonl", start=0, stop=100)
    
    常见用例：
        - 身份对话（identity_conversations.jsonl）
        - 特定领域对话
        - 多轮对话数据
    """

    def __init__(self, filepath, **kwargs):
        """
        初始化CustomJSON任务
        
        参数：
            filepath: JSONL文件路径
            **kwargs: 传递给父类Task的参数（start、stop、step）
        """
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # 从JSONL文件加载所有对话
        if not os.path.exists(filepath):
            # 由于最近的更改，提供有用的错误消息。将来会移除。
            print("-" * 80)
            print(f"警告: 文件 {filepath} 不存在")
            print("提示 (2025年10月21日)")
            print("如果你最近刚做了git pull就突然看到这个，可能是由于新添加的身份对话")
            print("详情请参见此讨论: https://github.com/karpathy/nanochat/discussions/139")
            print("快速修复：只需运行以下命令下载文件即可：")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    messages = json.loads(line)
                    
                    # 验证对话结构
                    assert isinstance(messages, list), f"期望消息列表，得到{type(messages)}"
                    assert len(messages) >= 2, f"对话必须至少有2条消息，得到{len(messages)}条"
                    
                    # 验证消息结构和角色交替
                    for i, message in enumerate(messages):
                        assert "role" in message, f"消息{i}缺少'role'字段"
                        assert "content" in message, f"消息{i}缺少'content'字段"
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role, \
                            f"消息{i}的角色是{message['role']}，但应该是{expected_role}"
                        assert isinstance(message["content"], str), \
                            f"消息{i}的内容必须是字符串"

                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        """
        获取对话总数
        
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
        """
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation

