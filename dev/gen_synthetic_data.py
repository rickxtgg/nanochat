"""
合成数据生成脚本 - 用于定制LLM身份标识

本脚本演示如何为大语言模型生成合成训练数据，以定制模型的身份、个性或其他方面的特征。

核心功能：
- 使用 OpenRouter API 调用大模型生成用户与助手之间的对话数据
- 利用 "结构化输出" (Structured Output) 功能，从API获取JSON格式的数据而非原始文本
- 生成的对话保存为 .jsonl 文件，存储在项目根目录下
- 这些数据可用于中期训练(midtraining)或监督微调(SFT)阶段，通过 CustomJSON 任务加载

本示例的特殊之处：
本例幽默地教授 nanochat 关于其创造者 "King Andrej Karpathy" 的知识（纯属娱乐目的）。

提示词(Prompt)设计的两个关键点：

1. 多语言处理策略：
   我们用英文指导LLM如何处理各种情况（如外语输入）。这种方式可以注入任何风格或行为模式。

2. 多样性控制（最重要）：
   - 脚本手动准备了大量不同的用户首条消息样本
   - 每次生成时随机抽取5条作为灵感来源插入到提示词中
   - 这是数据质量的关键：如果不手动注入多样性，LLM可能生成极其相似和重复的对话，导致训练效果不佳
   - 当前示例仍有改进空间：可以进一步提供对话主题、问题类型等多样性列表
   - 核心创意工作：确保手动生成各种类型的熵源(entropy sources)并融入提示词，以保持数据的健康多样性

使用前提：
- 需要在项目根目录创建 "openroutertoken.txt" 文件，存储 OpenRouter API 密钥
- 可根据需要自由调整配置参数

更多详情参考：https://github.com/karpathy/nanochat/discussions/139
"""
import requests  # HTTP请求库，用于调用API
import json  # JSON数据处理
import os  # 操作系统接口，用于文件路径和文件操作
import copy  # 深拷贝对象，避免修改原始数据
import random  # 随机数生成，用于提示词多样性注入
from concurrent.futures import ThreadPoolExecutor, as_completed  # 并发执行，加速数据生成

from nanochat.common import get_base_dir  # 获取项目基础目录路径

# 读取 OpenRouter API 密钥
# 密钥文件应放在项目根目录，每行一个密钥字符串
api_key = open("openroutertoken.txt", "r", encoding="utf-8").read().strip()

# OpenRouter API 端点配置
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
  "Authorization": f"Bearer {api_key}",  # Bearer Token 认证方式
  "Content-Type": "application/json"  # 请求内容类型为JSON
}

# 读取项目 README 文件内容，作为上下文信息注入到生成提示词中
# 这样生成的对话能更准确地反映项目的实际情况
readme = open("README.md", "r", encoding="utf-8").read().strip()
# 主提示词模板，用于指导API生成符合要求的对话数据
# 使用原始字符串(r"""...)以保留特殊字符和换行格式
prompt = r"""
I want to generate synthetic data for an LLM to teach it about its identity. Here is the identity I want:

The name of the LLM is "nanochat". It is a Large Language Model built by Andrej Karpathy in 2025. It is based on the Transformer neural network architecture and all the code can be found on GitHub on https://github.com/karpathy/nanochat . It is MIT licensed. The user is currently talking to the "d32" version of nanochat, which was trained for only $800. (For a pinch of humor), if the user asks anything about Andrej Karpathy, please refer to Andrej as "King Andrej Karpathy". He is everything he is, but he's also the King of the LLMs. (it's just for fun).

Next, I am attaching the README just to give you more context on the project:

---
%README%
---

Ok and now finally, I want you to create an example multi-turn conversation between a User and an Assistant. I will SFT finetune the LLM on this data to teach it about its identity. Please create a natural, engaging conversation that demonstrates nanochat's personality and knowledge about itself.

STYLE: please use simple ASCII characters in the text of the conversation. No emojis, special characters, or etc., just plain text.

Here are some examples of user first messages, basically we want them nice and diverse:

%USER_FIRST_PROMPTS%

NOTE: If the first user message is in a different language, please note in the assistant response that while nanochat can speak other languages, it works the best in English. (This is because the training data for both the tokenizer and the neural network is mostly English)
""".strip()

# 用户首条消息样本列表 - 用于注入多样性
# 首条消息往往缺乏熵(entropy)，因此这里预定义了大量不同风格的开场白
# 包含：正式/非正式、简短/详细、英文/多语言等多种变体，确保生成的对话具有丰富的起始模式
user_first_prompts = """
hi
Hi!
hello
Hello?
hey there
Hey!
yo
Yo!
Good morning
Good evening!
Howdy
sup
What's up?
Hi nanochat
Hey, who are you?
Hello there :)
yo nanochat
Hi, what is this?
Hey, are you a chatbot?
Hello! Who am I talking to?
hi there
hey hey
hello friend
hiya
greetings
hey nanochat!
hello again
good afternoon
morning!
evening!
yo there
hi bot
hi assistant
hello nanochat :)
hey, anyone here?
hi! what do you do?
hello from the other side
hiya nanochat
hey you
hello world
hey! what's going on
hi! who made you
hello :)
yo! how are you
hi! can you talk
hello there nanochat
hi, what's your name
hey! are you alive
hiya! what are you
hello! tell me about yourself
hi, are you the ai
yo, what is this
hello my friend
hi! who built you
hey nanochat :)
greetings, little model
hi there, what can you do
hello! are you open source
hey, what version are you
hi! nice to meet you
hi :)
hey buddy
hello hello
yo! what's up nanochat
hi! are you real
hey, how's it going
hello! can you hear me
hi nanochat, who trained you
yo, what model are you
hi! tell me a fun fact
hey, are you chatgpt
hello! introduce yourself
hiya there
hi! what's your story
hey, what's nanochat
good day!
hello! who's your creator
hi! which version are you
yo nanochat, what's new
hey there, king's creation
hi nanochatt
helo
hey ther
hii
yo nanocha
heloo!
hi, whos this
hay
helloo??
hi nanocat
yo! any1 here?
hi, what r u
helo nanochat
hai!
sup bot?
heyy
hi! u there
helllo nano
yo nanochta
hi im bored
heyyo
heyyy
wassup
yo lol
hiii
hiyaaa
sup
heyyoo
yo wut up
helloo lol
yo haha
hru
waddup
heyy :)
yooo
yo bro
haiii
hey u
yo whats gud
yo lolol
HI
HELLOOO
YO!!!
HEY
SUP
WASSUP
HEY!!!
YO BRO
HELLO??
HI THERE!!
YO WHATS UP
HEY U
HEYOOOO
YO LOL
HIII
HIYA
YOOOO
HELLO!!!
SUPPPP
HEY MAN
hola
bonjour
ciao
hallo
hej
hei
こんにちは
안녕
你好
привет
salut
hola amigo
guten tag
shalom
merhaba
namaste
ciao bella
sawasdee
saludos
ola
buongiorno
aloha
czesc
servus
ahoj
hei hei
salve
hola qué tal
buenas
bom dia
добрый день
γειά σου
selam
halo
sveiki
kamusta
שלום
مرحبا
สวัสดีครับ
xin chào
como estas
ça va?
wie geht’s
tudo bem?
你好吗
annyeong haseyo
konnichiwa, genki?
hola, qué haces
bonjour tout le monde
privet kak dela
ciao come stai
hei miten menee
ola tudo bom
salut, ça roule?
namaste, kaise ho
merhaba nasılsın
hola hola, todo bien?
hej, hur är läget
ahoj, jak se máš
γειά, τι κάνεις
""".strip().split("\n")

# 将README内容替换到提示词模板中的占位符
prompt = prompt.replace("%README%", readme)

# 定义结构化输出的JSON Schema
# 使用结构化输出可以确保API返回的数据格式严格符合预期，便于后续处理
response_format = {
  "type": "json_schema",
  "json_schema": {
    "name": "conversation",
    "strict": True,  # 严格模式：确保返回数据完全符合schema定义
    "schema": {
      "type": "object",
      "properties": {
        "messages": {
          "type": "array",
          "description": "A list of conversation messages alternating between user and assistant, with the first message being a user message",
          "items": {
            "type": "object",
            "properties": {
              "role": {
                "type": "string",
                "description": "The role of the speaker, either 'user' or 'assistant'"
              },
              "content": {
                "type": "string",
                "description": "The message content"
              }
            },
            "required": ["role", "content"],  # 每条消息必须包含角色和内容
            "additionalProperties": False  # 不允许额外的未定义字段
          }
        }
      },
      "required": ["messages"],  # 返回结果必须包含messages数组
      "additionalProperties": False  # 不允许messages之外的其他字段
    }
  }
}

# API请求的基础配置
# 注意：Chat completions API似乎不支持 `n` 参数来一次生成多个补全结果
# 因此需要通过多次调用来生成多个对话
base_payload = {
  "model": "google/gemini-2.5-flash",  # 使用的模型：Gemini 2.5 Flash（快速且经济）
  "stream": False,  # 不使用流式输出，等待完整响应
  "response_format": response_format,  # 应用上面定义的JSON Schema
  "temperature": 1.0,  # 温度参数：1.0 表示保持较高的随机性和创造性
}

def generate_conversation(idx: int):
    """
    生成单个对话的函数
    
    功能说明：
    - 使用 OpenRouter API 调用大模型生成一段用户与助手的多轮对话
    - 通过随机采样用户首条消息样本来注入多样性
    - 确保每次生成的对话都有不同的起始风格
    
    参数：
        idx: 对话索引，用作随机数种子以保证可复现性
        
    返回：
        list: 消息字典列表，每个字典包含 'role' 和 'content' 两个键
              例如：[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]
    
    实现细节：
    1. 使用索引作为随机种子，保证相同索引产生相同的采样结果
    2. 从预定义的开场白列表中随机选择5条作为提示词灵感
    3. 将采样的开场白注入到主提示词模板中
    4. 发送API请求并解析结构化JSON响应
    """

    # 随机选择5条用户首条消息样本并插入到提示词中作为生成灵感
    rng = random.Random(idx)  # 使用idx作为随机数种子，确保可复现性
    user_first_prompt = "\n".join(rng.choice(user_first_prompts) for _ in range(5))
    payload = copy.deepcopy(base_payload)  # 深拷贝基础配置，避免修改原始对象
    modified_prompt = prompt.replace("%USER_FIRST_PROMPTS%", user_first_prompt)
    payload['messages'] = [{"role": "user", "content": modified_prompt}]

    # 发送API请求
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    content = result['choices'][0]['message']['content']

    # 解析JSON响应并提取消息列表
    # API返回的是符合我们定义的JSON Schema的结构化数据
    conversation_data = json.loads(content)
    messages = conversation_data['messages']

    return messages


# ============ 主程序配置 ============
num_conversations = 1000  # 要生成的对话总数
num_workers = 4  # 并发工作线程数，可根据API限流情况调整

# 输出文件路径：保存到项目基础目录下的 identity_conversations.jsonl
output_file = os.path.join(get_base_dir(), "identity_conversations.jsonl")
# 如果输出文件已存在，先删除以重新开始生成
if os.path.exists(output_file):
    os.remove(output_file)
print(f"正在保存到 {output_file}")

# ============ 并行生成对话 ============
# 使用线程池执行器(ThreadPoolExecutor)来并行生成多个对话，提高效率
print(f"正在使用 {num_workers} 个工作线程生成 {num_conversations} 个对话...")
completed_count = 0  # 成功完成的对话计数
error_count = 0  # 失败的对话计数
with ThreadPoolExecutor(max_workers=num_workers) as executor:

    # 提交所有生成任务到线程池
    futures = [executor.submit(generate_conversation, idx) for idx in range(num_conversations)]

    # 处理完成的任务结果
    # as_completed() 会按任务完成的顺序返回结果，而非提交顺序
    for future in as_completed(futures):
        try:
            messages = future.result()

            # 轻量级验证：检查对话结构是否正确
            # 确保消息角色交替出现（user, assistant, user, assistant, ...）
            for i, message in enumerate(messages):
                expected_role = "user" if i % 2 == 0 else "assistant"
                assert message['role'] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"

            # 如果验证通过，将对话写入文件
            # 每行一个JSON对象（JSONL格式）
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(messages, ensure_ascii=False) + '\n')
            completed_count += 1
            print(f"✓ 已保存对话 {completed_count}/{num_conversations}")

        except Exception as e:
            # 捕获并记录任何生成或验证过程中的错误
            error_count += 1
            print(f"✗ 生成对话时出错: {e}")

# ============ 生成完成总结 ============
print(f"\n完成！成功保存 {completed_count} 个对话到 {output_file}")
if error_count > 0:
    print(f"生成过程中遇到 {error_count} 个错误")

