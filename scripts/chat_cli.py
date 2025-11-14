"""
命令行对话接口 - nanochat 交互式聊天

功能说明：
本脚本提供命令行界面，用于与训练好的对话模型进行实时交互。
支持多轮对话、对话历史保持、以及单次提示模式。

运行方式：

1. 交互式对话模式（默认）：
   python -m scripts.chat_cli -i sft
   说明：进入交互模式，可以进行多轮对话

2. 单次提示模式：
   python -m scripts.chat_cli -i sft -p "What is the capital of France?"
   说明：提供提示词，获得单次回答后退出

3. 自定义生成参数：
   python -m scripts.chat_cli -i sft -t 0.8 -k 100
   说明：设置温度和top-k参数

4. 加载特定检查点：
   python -m scripts.chat_cli -i mid -g d20 -s 5000
   说明：加载特定模型标签和步数的检查点

命令说明：
- 'quit' 或 'exit': 退出对话
- 'clear': 清空对话历史，开始新对话
- 直接输入文本: 发送消息给模型

技术特性：
- 流式输出：实时显示模型生成的文本
- 上下文保持：维护完整的对话历史
- 特殊token处理：正确处理用户和助手的分隔标记
- 支持多种设备：CUDA、CPU、MPS

注意：
目前设计为单GPU运行，不支持分布式推理。
"""
import argparse  # 命令行参数解析
import torch  # PyTorch深度学习框架
from nanochat.common import compute_init, autodetect_device_type  # 通用工具函数
from contextlib import nullcontext  # 空上下文管理器
from nanochat.engine import Engine  # 文本生成引擎
from nanochat.checkpoint_manager import load_model  # 模型检查点加载

# =============================================================================
# 命令行参数配置
# =============================================================================
parser = argparse.ArgumentParser(description='与 nanochat 模型进行对话')
parser.add_argument('-i', '--source', type=str, default="sft", help="模型来源：sft|mid|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='要加载的模型标签')
parser.add_argument('-s', '--step', type=int, default=None, help='要加载的训练步数')
parser.add_argument('-p', '--prompt', type=str, default='', help='单次提示模式：提供提示词，获得回答后退出')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='生成温度参数（控制随机性）')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k采样参数（限制候选词数量）')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='设备类型：cuda|cpu|mps（空值表示自动检测）')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'], help='数据类型')
args = parser.parse_args()

# =============================================================================
# 初始化模型和分词器
# =============================================================================
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
# 设置混合精度上下文（CUDA使用指定精度）
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
# 加载模型检查点
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# =============================================================================
# 对话状态机的特殊token
# =============================================================================
# 这些特殊token用于标记对话的不同部分
bos = tokenizer.get_bos_token_id()  # 对话开始标记
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")  # 用户消息边界
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")  # 助手消息边界

# =============================================================================
# 创建生成引擎
# =============================================================================
engine = Engine(model, tokenizer)

# =============================================================================
# 显示欢迎信息
# =============================================================================
print("\nNanoChat 交互模式")
print("-" * 50)
print("输入 'quit' 或 'exit' 结束对话")
print("输入 'clear' 开始新对话")
print("-" * 50)

# 初始化对话token序列（从BOS开始）
conversation_tokens = [bos]

# =============================================================================
# 主对话循环
# =============================================================================
while True:

    # =========================================================================
    # 获取用户输入
    # =========================================================================
    if args.prompt:
        # 单次提示模式：从命令行参数获取提示词
        user_input = args.prompt
    else:
        # 交互模式：从控制台获取用户输入
        try:
            user_input = input("\n用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            # 处理 Ctrl+D 或 Ctrl+C
            print("\n再见！")
            break

    # =========================================================================
    # 处理特殊命令
    # =========================================================================
    if user_input.lower() in ['quit', 'exit']:
        # 退出命令
        print("再见！")
        break

    if user_input.lower() == 'clear':
        # 清空对话历史，重新开始
        conversation_tokens = [bos]
        print("对话已清空。")
        continue

    if not user_input:
        # 空输入，继续等待
        continue

    # =========================================================================
    # 将用户消息添加到对话历史
    # =========================================================================
    conversation_tokens.append(user_start)  # 用户消息开始标记
    conversation_tokens.extend(tokenizer.encode(user_input))  # 用户消息内容
    conversation_tokens.append(user_end)  # 用户消息结束标记

    # =========================================================================
    # 生成助手回复
    # =========================================================================
    conversation_tokens.append(assistant_start)  # 助手消息开始标记
    generate_kwargs = {
        "num_samples": 1,  # 生成1个样本
        "max_tokens": 256,  # 最多生成256个token
        "temperature": args.temperature,  # 温度参数（控制随机性）
        "top_k": args.top_k,  # Top-k采样参数
    }
    response_tokens = []
    print("\n助手: ", end="", flush=True)
    
    # 流式生成并显示
    with autocast_ctx:
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0]  # 移除批次维度（num_samples=1）
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)  # 实时打印token
    print()  # 换行
    
    # 确保助手消息以结束标记结尾
    # 即使由于达到max_tokens而停止生成，也要追加结束标记
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    
    # 将助手回复添加到对话历史
    conversation_tokens.extend(response_tokens)

    # =========================================================================
    # 单次提示模式：获得回答后退出
    # =========================================================================
    if args.prompt:
        break
