"""
任务（Task）基类

Task是所有任务的基类。一个Task本质上是一个对话数据集，
附带一些元数据和通常还有评估标准。

任务示例：
    - MMLU: 大规模多任务语言理解
    - ARC-Easy/Challenge: AI2推理挑战赛
    - GSM8K: 小学数学应用题
    - HumanEval: Python代码生成
    - SmolTalk: 对话数据集

设计理念：
    - 统一接口：所有任务都实现相同的接口
    - 轻量切片：支持对底层数据集的逻辑视图（start/stop/step）
    - 评估分离：每个任务定义自己的评估逻辑
    - 可组合：支持任务混合（TaskMixture）和序列（TaskSequence）
"""

import random  # 用于任务混合时的确定性打乱

class Task:
    """
    任务基类
    
    功能：
        - 提供统一的数据集接口
        - 支持轻量级切片（不复制底层数据）
        - 定义评估接口
    
    子类必须实现的方法：
        - eval_type: 返回评估类型（'generative'或'categorical'）
        - num_examples: 返回数据集总样本数
        - get_example: 获取指定索引的样本
        - evaluate: 评估assistant的回答
    
    切片参数：
        - start: 起始索引（默认0）
        - stop: 结束索引（默认None=到末尾）
        - step: 步长（默认1）
    
    用法示例：
        # 获取数据集的前100个样本
        task = SomeTask(start=0, stop=100)
        
        # 每隔2个取一个样本
        task = SomeTask(start=0, stop=1000, step=2)
    """

    def __init__(self, start=0, stop=None, step=1):
        """
        初始化任务
        
        参数：
            start: 起始索引（必须>=0）
            stop: 结束索引（None表示到末尾，否则必须>=start）
            step: 步长（必须>=1）
        
        这允许对数据集进行轻量级的逻辑视图，不复制底层数据
        """
        assert start >= 0, f"起始索引必须非负，得到{start}"
        assert stop is None or stop >= start, f"结束索引应该大于等于起始索引，得到{stop}和{start}"
        assert step >= 1, f"步长必须严格为正，得到{step}"
        self.start = start
        self.stop = stop  # 这里可以是None
        self.step = step

    @property
    def eval_type(self):
        """
        评估类型
        
        返回：
            'generative': 生成式任务（如GSM8K、HumanEval）
            'categorical': 分类任务（如MMLU、ARC）
        
        子类必须实现此方法
        """
        raise NotImplementedError

    def num_examples(self):
        """
        获取底层数据集的总样本数
        
        返回：
            样本总数（整数）
        
        子类必须实现此方法
        """
        raise NotImplementedError

    def get_example(self, index):
        """
        获取指定物理索引的样本
        
        参数：
            index: 物理索引（在底层数据集中的真实索引）
        
        返回：
            conversation: 对话字典
        
        子类必须实现此方法
        """
        raise NotImplementedError

    def __len__(self):
        """
        获取切片后的样本数量
        
        返回：
            切片视图中的样本数量
        
        计算方式：
            span = stop - start
            num = ceil(span / step)
        """
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step  # ceil_div(span, step)
        assert num >= 0, f"样本数量为负？？？: {num}"  # 防止错误
        return num

    def __getitem__(self, index: int):
        """
        通过逻辑索引获取样本
        
        参数：
            index: 逻辑索引（在切片视图中的索引）
        
        返回：
            conversation: 对话字典
        
        逻辑索引到物理索引的转换：
            physical_index = start + index * step
        """
        assert isinstance(index, int), f"索引必须是整数，得到{type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        """
        评估assistant的回答
        
        参数：
            problem: 问题/对话
            completion: assistant的回答
        
        返回：
            评估结果（具体类型取决于子类）
        
        子类必须实现此方法
        """
        raise NotImplementedError


class TaskMixture(Task):
    """
    任务混合类（用于SFT训练）
    
    用途：
        在SFT训练中，混合多个数据集进行训练非常有用。
        这可以提高模型在多个任务上的泛化能力。
    
    特性：
        1. 确定性打乱：使用固定种子（42）打乱所有样本
        2. 任务混合：确保不同任务的样本在训练中均匀分布
        3. 过采样技巧：如果想过采样某个任务，只需在列表中多次传入
    
    工作原理：
        - 构建所有(task_idx, local_idx)对的列表
        - 确定性打乱这个列表
        - 访问时根据打乱后的索引映射获取样本
    
    用法示例：
        # 混合多个任务
        mixture = TaskMixture([task1, task2, task3])
        
        # 过采样task1（出现2次）
        mixture = TaskMixture([task1, task1, task2, task3])
    
    注意：
        这不是最优雅或最好的解决方案，但目前来说足够用了
    """

    def __init__(self, tasks, **kwargs):
        """
        初始化任务混合
        
        参数：
            tasks: Task对象列表
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        # tasks是Task对象的列表
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        
        # 构建所有(task_idx, local_idx)对的列表
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        
        # 确定性打乱以在整个训练过程中混合任务
        rng = random.Random(42)
        rng.shuffle(self.index_map)

    def num_examples(self):
        """获取总对话数"""
        return self.num_conversations

    def get_example(self, index):
        """
        根据确定性打乱的所有样本访问对话
        
        这确保任务在训练过程中混合，无论数据集大小如何
        
        参数：
            index: 全局索引
        
        返回：
            conversation: 对话字典
        """
        assert 0 <= index < self.num_conversations, \
            f"索引{index}超出范围，混合数据集有{self.num_conversations}个对话"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """
    任务序列类（用于课程学习）
    
    用途：
        在SFT训练中，有时我们想要按顺序训练任务列表。
        这对需要训练课程（curriculum learning）的情况很有用。
    
    特性：
        - 顺序访问：按任务顺序依次访问样本
        - 无打乱：保持原始任务顺序
        - 课程学习：可以从简单任务到困难任务
    
    工作原理：
        - 按顺序拼接所有任务
        - 访问时先定位到对应的任务，再获取该任务中的样本
    
    用法示例：
        # 按顺序训练：先ARC-Easy，再ARC-Challenge
        sequence = TaskSequence([arc_easy, arc_challenge])
        
        # 课程学习：从简单到困难
        sequence = TaskSequence([easy_task, medium_task, hard_task])
    
    与TaskMixture的区别：
        - TaskMixture: 打乱所有任务的样本，均匀混合
        - TaskSequence: 保持任务顺序，依次训练
    """

    def __init__(self, tasks, **kwargs):
        """
        初始化任务序列
        
        参数：
            tasks: Task对象列表（按顺序）
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        """获取总对话数"""
        return self.num_conversations

    def get_example(self, index):
        """
        按顺序获取样本
        
        参数：
            index: 全局索引
        
        返回：
            conversation: 对话字典
        
        查找逻辑：
            遍历任务列表，从index中减去每个任务的长度，
            直到找到包含该索引的任务
        """
        assert 0 <= index < self.num_conversations, \
            f"索引{index}超出范围，序列有{self.num_conversations}个对话"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def render_mc(question, letters, choices):
    """
    渲染多选题的通用格式
    
    参数：
        question: 问题文本
        letters: 选项字母列表（如["A", "B", "C", "D"]）
        choices: 选项内容列表（如["选项1", "选项2", "选项3", "选项4"]）
    
    返回：
        格式化的多选题提示字符串
    
    格式：
        Multiple Choice question: {question}
        - {choice1}=A
        - {choice2}=B
        - {choice3}=C
        - {choice4}=D
        
        Respond only with the letter of the correct answer.
    
    重要的设计决策：
    
    1. 字母在选项之后：
       更大的模型不太在意，但较小的模型更喜欢将字母放在选项*之后*，
       这会导致更好的绑定（binding）。
    
    2. 分隔符(=)和字母之间没有空格：
       这实际上很关键，因为分词器对" A"和"A"有不同的token ID。
       assistant的回答将只是字母本身，即"A"，所以在提示中使用
       完全相同的token很重要，即"A"前面没有空格。
       同样，更大的模型对此不太在意，但较小的模型确实关心这些细节。
    
    示例：
        >>> render_mc("天空是什么颜色？", ["A", "B", "C"], ["红色", "蓝色", "绿色"])
        Multiple Choice question: 天空是什么颜色？
        - 红色=A
        - 蓝色=B
        - 绿色=C
        
        Respond only with the letter of the correct answer.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


if __name__ == "__main__":
    """
    轻量级切片测试
    
    测试Task类的切片功能是否正常工作
    """
    from tasks.mmlu import MMLU

    # 测试完整数据集
    ds = MMLU(subset="auxiliary_train", split="train")
    print("MMLU长度: ", len(ds))
    ex = ds[5]
    print("第5个样本: ", ex)

    # 测试切片
    ds = MMLU(subset="auxiliary_train", split="train", start=5, stop=10)
    print("切片后的MMLU[5:10]长度: ", len(ds))
    print("切片后的第0个样本: ", ds[0])

    # 验证切片是否正确
    print("两者匹配: ", ex == ds[0])
