"""
拼写蜜蜂任务（SpellingBee）- 提高模型的拼写和计数能力

这个任务旨在让nanochat在拼写和计数方面表现更好，例如：
    问题："How many r are in strawberry?"
    答案：3

任务设计理念：
    这个任务的一个有趣之处是，我们会让assistant使用手动计数和Python的组合来解决问题。
    这是一个很好的问题解决"本能"，可以混合到模型中，RL可能会进一步优化它，
    让模型学会何时信任手动计数，何时信任Python。
    
    如果我们更高级（我们可以/应该这样做），我们会在这里和那里添加一些小错误，
    让模型也学会从错误中恢复。我们可以在未来的版本中做到这一点。

本文件包含两个任务：
    1. SpellingBee: 计算单词中某个字母的出现次数
    2. SimpleSpelling: 简单地拼写单词

设计动机：
    (1)是目标任务，但(2)存在是因为它是(1)困难部分的高度浓缩版本，
    即单词拼写。这对LLM来说并非易事，因为它必须学习每个token（一个小的语义块/原子）
    如何映射到组成它的单个字符序列。
    
    更大的模型最终会自己学会这一点，但如果我们希望这种能力存在于较小的模型中，
    我们必须通过在训练数据中过度表示它来积极鼓励它。Midtraining是做这件事的好地方。

多语言支持：
    本任务包含多种语言的提示变体（英语、西班牙语、中文、韩语、法语、德语、日语），
    增强模型的多语言理解能力。

预览示例：
    运行以下命令查看生成的对话示例：
    python -m tasks.spellingbee
"""

import re  # 正则表达式（用于提取答案）
import random  # 随机选择（用于数据增强）
from tasks.common import Task  # 任务基类
from nanochat.common import download_file_with_lock  # 带锁的文件下载

# 26个英文字母
LETTERS = "abcdefghijklmnopqrstuvwxyz"

# 370K英文单词列表（品种多样）
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"

# 答案提取正则表达式（与GSM8K相同）
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    """
    提取#### 标记后的数值答案
    
    参数：
        completion: 完整的回答文本
    
    返回：
        提取的数值答案（字符串），如果未找到则返回None
    
    注意：
        此函数与GSM8K的答案提取完全相同
    """
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")  # 移除千位分隔符
        return match_str
    return None

# 用户消息模板（用于数据增强）
# 包含多种语言和多种表达方式，增加模型对不同表述的鲁棒性
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # 西班牙语提示
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # 简体中文提示
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # 韩语提示
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # 法语提示
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # 德语提示
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # 日语提示
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]

class SpellingBee(Task):
    """
    拼写蜜蜂任务类
    
    目标：
        训练模型计算单词中特定字母的出现次数，
        并学会结合手动计数和Python工具来解决问题。
    
    特性：
        - 动态生成：基于370K单词列表
        - 数据增强：多种语言和表达方式
        - 工具使用：结合手动计数和Python验证
        - 可复现：使用确定性随机种子
    
    训练策略：
        1. 手动拼写单词（培养字符级理解）
        2. 逐个字符计数（培养逻辑推理）
        3. Python验证（培养工具使用）
        4. 最终答案（培养答案格式）
    
    用法示例：
        # 创建训练集
        task = SpellingBee(size=1000, split="train")
        
        # 创建测试集
        task = SpellingBee(size=100, split="test")
    """

    def __init__(self, size=1000, split="train", **kwargs):
        """
        初始化SpellingBee任务
        
        参数：
            size: 生成的样本数量（默认1000）
            split: 数据分割，必须是"train"或"test"
            **kwargs: 传递给父类Task的参数
        """
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee分割必须是train或test"
        self.size = size
        self.split = split
        # 下载370K单词列表
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        self.words = words

    @property
    def eval_type(self):
        """评估类型：生成式任务"""
        return 'generative'

    def num_examples(self):
        """返回样本数量"""
        return self.size

    def get_example(self, index):
        """
        生成指定索引的样本
        
        参数：
            index: 样本索引
        
        返回：
            conversation: 对话字典
        
        生成流程：
            1. 使用确定性随机种子（基于index和split）
            2. 随机选择单词
            3. 选择字母（90%来自单词，10%随机）
            4. 计算正确答案
            5. 生成多样化的用户消息
            6. 构建详细的assistant回答（手动计数+Python验证）
        """
        # 避免train和test在index=0时碰撞
        seed = index if self.split == "train" else -(index + 1)
        rng = random.Random(seed)

        # 随机选择一个单词
        word = rng.choice(self.words)
        # 90%从单词中选择字母，10%随机选择字母
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)

        # 简单计数获取正确答案
        count = word.count(letter)

        # 创建用户消息，使用多种变体进行数据增强
        template = rng.choice(USER_MSG_TEMPLATES)
        # 30%的概率全小写（模拟懒人不按Shift）
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ['', "'", '"']
        letter_quote = rng.choice(quote_options)  # 字母是否加引号？
        word_quote = rng.choice(quote_options)    # 单词是否加引号？
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5:  # 50%的人甚至不用问号
            user_msg += "?"

        # 现在创建理想的assistant回答 - 构建为parts（文本+工具调用）
        assistant_parts = []
        word_letters = ",".join(list(word))
        manual_text = f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""
        # 模拟解决过程的小循环
        # TODO: 这里是有趣的地方，我们可以模拟一些可爱的小错误
        # 让模型学会检查其工作并从错误中恢复。
        # 你当然可以希望这也能在RL中出现，但实际上你会想帮助它一点。
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                # 注意：这里i和char之间故意不能有空格
                # 因为这会创建不同的token！（例如" a"和"a"是不同的token）
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"

        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})
        # 第2部分：Python验证
        assistant_parts.append({"type": "text", "text": "\n\nLet me double check this using Python:\n\n"})
        # 第3部分：Python工具调用
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        # 第4部分：Python输出
        assistant_parts.append({"type": "python_output", "text": str(count)})
        # 第5部分：最终答案
        assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"})

        # 返回完整对话
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        评估assistant的回答是否正确
        
        参数：
            conversation: 对话字典（包含ground truth）
            assistant_response: assistant的回答
        
        返回：
            int: 0表示错误，1表示正确
        
        注意：
            此评估逻辑与GSM8K完全相同
        """
        assert isinstance(assistant_response, str), "目前假设是简单字符串回答"
        # 首先从对话中提取ground truth答案
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "最后一条消息必须来自Assistant"
        assert isinstance(assistant_message['content'], list), "期望这是一个parts列表"
        # 最后一个text part包含带####的最终答案
        last_text_part = assistant_message['content'][-1]['text']
        # 提取ground truth答案和预测答案
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # 比较并返回成功结果（整数）
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        用于RL的奖励函数
        
        使用简单的0-1奖励，就像GSM8K一样
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float


class SimpleSpelling(Task):
    """
    简单拼写任务类
    
    目标：
        让模型练习拼写单词的更简单任务。
        这是SpellingBee任务中最困难部分（单词拼写）的浓缩版本。
    
    设计动机：
        对LLM来说，学习token到字符序列的映射并非易事。
        这个任务通过过度表示单词拼写来帮助较小模型学习这一能力。
    
    任务格式：
        用户：Spell the word: {word}
        助手：{word}:{逗号分隔的字母}
    
    用法示例：
        # 创建训练集
        task = SimpleSpelling(size=5000, split="train")
    """

    def __init__(self, size=1000, split="train", **kwargs):
        """
        初始化SimpleSpelling任务
        
        参数：
            size: 生成的样本数量（默认1000）
            split: 数据分割，必须是"train"或"test"
            **kwargs: 传递给父类Task的参数
        """
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee分割必须是train或test"
        self.size = size
        self.split = split
        # 下载370K单词列表
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words)  # 使用与SpellingBee任务不同的单词顺序
        self.words = words

    @property
    def eval_type(self):
        """评估类型：生成式任务"""
        return 'generative'

    def num_examples(self):
        """返回样本数量"""
        return self.size

    def get_example(self, index):
        """
        生成指定索引的样本
        
        参数：
            index: 样本索引
        
        返回：
            conversation: 对话字典
        
        格式：
            用户：Spell the word: {word}
            助手：{word}:{逗号分隔的字母}
        """
        # 避免train和test在index=0时碰撞
        seed = index if self.split == "train" else -(index + 1)
        rng = random.Random(seed)
        # 随机选择一个单词
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        # 返回完整对话
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation


if __name__ == "__main__":
    """
    预览SpellingBee任务的前10个样本
    
    运行方式：
        python -m tasks.spellingbee
    """

    # 预览SpellingBee任务的前10个示例
    task = SpellingBee()
    for i in range(10):
        ex = task.get_example(i)
        print("=" * 100)
        print(ex['messages'][0]['content'])
        print("-" * 100)
        # Assistant内容现在是parts列表
        assistant_parts = ex['messages'][1]['content']
        for part in assistant_parts:
            if part['type'] == 'text':
                print(part['text'], end='')
            elif part['type'] == 'python':
                print(f"<<{part['text']}=", end='')
            elif part['type'] == 'python_output':
                print(f"{part['text']}>>", end='')
        print()
        print("-" * 100)

    # # 预览SimpleSpelling任务的前10个示例（已注释）
    # task = SimpleSpelling()
    # for i in range(10):
    #     ex = task.get_example(i)
    #     print("=" * 100)
    #     print(ex['messages'][0]['content'])
    #     print("-" * 100)
    #     print(ex['messages'][1]['content'])

    # # 同时检查分词（仅最后一个示例）（已注释）
    # from nanochat.tokenizer import get_tokenizer
    # tokenizer = get_tokenizer()
    # ids, mask = tokenizer.render_conversation(ex)
    # print(tokenizer.visualize_tokenization(ids, mask, with_token_id=True))
