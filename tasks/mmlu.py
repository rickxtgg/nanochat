"""
MMLU数据集（大规模多任务语言理解）

MMLU（Massive Multitask Language Understanding）是一个综合性的多项选择题数据集，
涵盖57个学科，包括STEM、人文、社会科学等，用于测试模型的知识广度和深度。

数据集特点：
    - 57个不同学科（从小学数学到专业法律）
    - 约15,000道多项选择题
    - 每题4个选项（A、B、C、D）
    - 测试从基础知识到专家级知识
    - auxiliary_train子集用于few-shot学习

数据来源：
    https://huggingface.co/datasets/cais/mmlu

学科分类：
    - STEM: 数学、物理、化学、计算机科学等
    - 人文: 历史、哲学、法律等
    - 社会科学: 经济学、心理学、社会学等
    - 其他: 商业、医学、营养学等

参考文献：
    Measuring Massive Multitask Language Understanding
    https://arxiv.org/abs/2009.03300
"""

from datasets import load_dataset  # HuggingFace datasets库
from tasks.common import Task, render_mc  # 任务基类和多选题渲染函数

class MMLU(Task):
    """
    MMLU数据集任务类
    
    特性：
        - 评估类型：categorical（分类/多选题）
        - 57个学科领域
        - 每题4个选项（A、B、C、D）
        - 支持auxiliary_train子集（用于few-shot）
    
    类属性：
        letters: 选项字母元组（'A', 'B', 'C', 'D'）
        groups: 57个学科名称元组
    
    用法示例：
        # 加载MMLU所有学科的测试集
        task = MMLU(subset="all", split="test")
        
        # 加载auxiliary_train用于few-shot
        task = MMLU(subset="auxiliary_train", split="train")
        
        # 加载验证集的前100个样本
        task = MMLU(subset="all", split="validation", start=0, stop=100)
    """

    # 选项字母
    letters = ('A', 'B', 'C', 'D')
    
    # 57个学科组（从抽象代数到世界宗教）
    groups = ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions')

    def __init__(self, subset, split, **kwargs):
        """
        初始化MMLU任务
        
        参数：
            subset: 子集名称，必须是"all"或"auxiliary_train"
            split: 数据分割，必须是"train"、"validation"、"dev"或"test"
            **kwargs: 传递给父类Task的参数（start、stop、step）
        
        注意：
            - subset="all": 包含所有57个学科
            - subset="auxiliary_train": 用于few-shot学习的辅助训练集
            - auxiliary_train只能使用split="train"
        """
        super().__init__(**kwargs)
        assert subset in ["all", "auxiliary_train"], f"子集{subset}必须是all或auxiliary_train"
        assert split in ["train", "validation", "dev", "test"], f"分割{split}必须是train、validation、dev或test"
        if subset == "auxiliary_train":
            assert split == "train", "auxiliary_train必须使用split=train"
        self.subset = subset
        self.split = split
        # 加载数据集并打乱（使用固定种子以保证可复现性）
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)
        if subset == "auxiliary_train":
            # 不明白为什么，但auxiliary_train的行有一个奇怪的额外'train'包装
            # 需要展开这个包装
            self.ds = self.ds.map(lambda row: row['train'], remove_columns=['train'])

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
                - subject: 学科名称（用于按学科分组指标）
                - letters: 选项字母列表（用于评估时限制预测范围）
        
        数据格式：
            - question: 问题文本
            - choices: 4个选项的文本列表
            - answer: 答案索引（0,1,2,3对应A,B,C,D）
            - subject: 学科名称（如"college_biology"）
        """
        row = self.ds[index]
        question = row["question"]  # 问题文本
        choices = row["choices"]  # 每个选项的文本
        answer = row["answer"]  # 答案的索引，例如0,1,2,3（对应A,B,C,D）
        subject = row["subject"]  # 例如"college_biology"、"college_chemistry"等
        assert len(choices) == 4, "MMLU应该有4个选项"
        
        # 创建并返回对话对象
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "subject": subject,  # 以后可能用于按学科分组指标
            "letters": self.letters,  # 在评估时有用，可以将assistant的预测限制到这些字母之一
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
        assert assistant_response in self.letters, \
            f"MMLU答案{assistant_response}应该是{self.letters}之一"
        assistant_message = conversation['messages'][-1]['content']  # 例如"A"
        return assistant_response == assistant_message
