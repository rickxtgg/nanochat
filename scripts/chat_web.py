#!/usr/bin/env python3
"""
NanoChat 统一Web聊天服务器 - 从单个FastAPI实例同时提供UI和API服务

架构设计：
本服务器使用数据并行（Data Parallelism）将请求分发到多个GPU上。
每个GPU加载完整的模型副本，incoming requests被分配到可用的worker上，
实现高并发和负载均衡。

核心特性：
1. 多GPU支持：可配置使用1-8个GPU，自动负载均衡
2. Worker池管理：每个GPU对应一个worker，通过asyncio队列调度
3. 流式响应：实时生成和传输token，提升用户体验
4. 滥用防护：限制消息长度、对话长度和生成参数
5. 健康监控：提供worker状态和GPU使用率统计接口
6. CORS支持：允许跨域访问API

运行示例：

1. 单GPU模式（默认）：
   python -m scripts.chat_web
   说明：适用于开发和轻量级部署

2. 多GPU模式（4卡）：
   python -m scripts.chat_web --num-gpus 4
   说明：生产环境推荐，显著提升并发能力

3. 自定义配置：
   python -m scripts.chat_web --num-gpus 2 --port 8080 --temperature 1.0
   说明：指定GPU数量、端口和默认生成参数

4. 使用特定模型：
   python -m scripts.chat_web --source rl --model-tag d12 --step 5000
   说明：加载RL训练的特定检查点

访问方式：
启动后，在浏览器打开控制台打印的URL（如果在云服务器上，确保使用公网IP）

API端点：
  GET  /              - 聊天UI（用户界面）
  POST /chat/completions - 聊天API（仅支持流式响应）
  GET  /health        - 健康检查，包含worker池状态
  GET  /stats         - Worker池统计和GPU利用率

滥用防护措施：
  - 单次请求最多500条消息
  - 单条消息最多8000字符
  - 对话总长度最多32000字符
  - Temperature范围限制在0.0-2.0
  - Top-k范围限制在1-200
  - Max tokens范围限制在1-4096
"""

import argparse  # 命令行参数解析
import json  # JSON处理
import os  # 操作系统接口
import torch  # PyTorch深度学习框架
import asyncio  # 异步I/O
import logging  # 日志记录
import random  # 随机数生成
from contextlib import asynccontextmanager  # 异步上下文管理器
from fastapi import FastAPI, HTTPException  # FastAPI Web框架
from fastapi.middleware.cors import CORSMiddleware  # CORS跨域中间件
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse  # 响应类型
from pydantic import BaseModel  # 数据验证模型
from typing import List, Optional, AsyncGenerator  # 类型提示
from dataclasses import dataclass  # 数据类装饰器
from contextlib import nullcontext  # 空上下文管理器
from nanochat.common import compute_init, autodetect_device_type  # 通用工具
from nanochat.checkpoint_manager import load_model  # 模型加载
from nanochat.engine import Engine  # 文本生成引擎

# =============================================================================
# 滥用防护限制
# =============================================================================
# 这些限制防止恶意用户过度消耗服务器资源
MAX_MESSAGES_PER_REQUEST = 500  # 单次请求最多消息数
MAX_MESSAGE_LENGTH = 8000  # 单条消息最大字符数
MAX_TOTAL_CONVERSATION_LENGTH = 32000  # 对话总字符数上限
MIN_TEMPERATURE = 0.0  # 最小温度
MAX_TEMPERATURE = 2.0  # 最大温度
MIN_TOP_K = 1  # 最小top-k
MAX_TOP_K = 200  # 最大top-k
MIN_MAX_TOKENS = 1  # 最小生成token数
MAX_MAX_TOKENS = 4096  # 最大生成token数

# =============================================================================
# 命令行参数配置
# =============================================================================
parser = argparse.ArgumentParser(description='NanoChat Web Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='使用的GPU数量（默认：1）')
parser.add_argument('-i', '--source', type=str, default="sft", help="模型来源：sft|mid|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='默认生成温度')
parser.add_argument('-k', '--top-k', type=int, default=50, help='默认top-k采样参数')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='默认最大生成token数')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='模型标签')
parser.add_argument('-s', '--step', type=int, default=None, help='加载的步数')
parser.add_argument('-p', '--port', type=int, default=8000, help='服务器端口')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'], help='数据类型')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='设备类型：cuda|cpu|mps（空值=自动检测）')
parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器绑定地址')
args = parser.parse_args()

# =============================================================================
# 日志配置
# =============================================================================
# 记录对话流量和服务器事件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 设备和精度初始化
# =============================================================================
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

# =============================================================================
# Worker 和 WorkerPool 类定义
# =============================================================================

@dataclass
class Worker:
    """
    Worker - 单个GPU上的模型实例
    
    每个Worker包含：
    - gpu_id: GPU编号
    - device: torch设备对象
    - engine: 文本生成引擎
    - tokenizer: 分词器
    - autocast_ctx: 混合精度上下文（用于加速推理）
    """
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    autocast_ctx: torch.amp.autocast

class WorkerPool:
    """
    Worker池 - 管理多个Worker实例的负载均衡
    
    架构设计：
    - 每个GPU上加载一个完整的模型副本
    - 使用asyncio.Queue实现Worker的分配和回收
    - 请求到达时从池中获取可用Worker，处理完成后归还
    - 自动实现负载均衡，充分利用多GPU资源
    """

    def __init__(self, num_gpus: Optional[int] = None):
        """
        初始化Worker池
        
        参数：
            num_gpus: GPU数量（None表示自动检测）
        """
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1  # CPU或MPS模式
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """
        初始化Worker池 - 在每个GPU上加载模型
        
        参数：
            source: 模型来源（sft|mid|rl）
            model_tag: 模型标签
            step: 检查点步数
        """
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "只有CUDA支持多GPU。CPU/MPS不支持。"

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type)  # CPU或MPS
                print(f"Loading model on {device_type}...")

            # 在指定设备上加载模型
            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

            # 创建Worker并添加到池中
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
                autocast_ctx=autocast_ctx
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)  # 标记为可用

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """
        从池中获取一个可用的Worker
        
        如果所有Worker都在忙，会等待直到有Worker可用
        """
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """
        将Worker归还到池中
        """
        await self.available_workers.put(worker)

# =============================================================================
# 数据模型定义（Pydantic）
# =============================================================================

class ChatMessage(BaseModel):
    """单条聊天消息"""
    role: str  # "user" 或 "assistant"
    content: str  # 消息内容

class ChatRequest(BaseModel):
    """聊天请求"""
    messages: List[ChatMessage]  # 消息列表
    temperature: Optional[float] = None  # 生成温度（可选）
    max_tokens: Optional[int] = None  # 最大生成token数（可选）
    top_k: Optional[int] = None  # Top-k采样参数（可选）

# =============================================================================
# 请求验证函数
# =============================================================================

def validate_chat_request(request: ChatRequest):
    """
    验证聊天请求，防止滥用
    
    检查项目：
    1. 消息数量（1-500条）
    2. 单条消息长度（最多8000字符）
    3. 对话总长度（最多32000字符）
    4. 生成参数范围（temperature、top_k、max_tokens）
    """
    # 检查消息数量
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="至少需要一条消息")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"消息过多。最多允许{MAX_MESSAGES_PER_REQUEST}条消息"
        )

    # 检查单条消息长度和对话总长度
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"消息{i}内容为空")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"对话总长度过长。最多允许{MAX_TOTAL_CONVERSATION_LENGTH}字符"
        )

    # 验证角色值
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"消息{i}角色无效。必须是'user'或'assistant'"
            )

    # 验证temperature参数
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature必须在{MIN_TEMPERATURE}和{MAX_TEMPERATURE}之间"
            )

    # 验证top_k参数
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k必须在{MIN_TOP_K}和{MAX_TOP_K}之间"
            )

    # 验证max_tokens参数
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens必须在{MIN_MAX_TOKENS}和{MAX_MAX_TOKENS}之间"
            )

# =============================================================================
# FastAPI 应用初始化
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    在启动时：加载所有GPU上的模型
    在关闭时：清理资源（通过yield后的代码，本例中无需显式清理）
    """
    print("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"Server ready at http://localhost:{args.port}")
    yield  # 应用运行期间
    # 这里可以添加清理代码（如果需要）

app = FastAPI(lifespan=lifespan, title="NanoChat API", version="1.0")

# 添加CORS中间件（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源（生产环境应限制）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# =============================================================================
# API 端点定义
# =============================================================================

@app.get("/")
async def root():
    """
    GET / - 提供聊天UI界面
    
    返回嵌入式HTML UI，用户可以在浏览器中直接聊天
    """
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # 替换API_URL为相对路径（使用相同的origin）
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """GET /logo.svg - 提供NanoChat logo（用于favicon和页头）"""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

# =============================================================================
# 核心生成函数
# =============================================================================

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """
    流式生成助手回复
    
    关键技术：
    1. UTF-8处理：正确处理多字节字符（如emoji、中文）
    2. 增量解码：累积token并逐步解码，避免输出乱码
    3. 替换字符检测：检测到'�'时等待更多token，确保完整UTF-8序列
    4. Server-Sent Events (SSE)：使用SSE协议实现实时流式传输
    
    参数：
        worker: Worker实例（包含模型和分词器）
        tokens: 输入token序列
        temperature: 生成温度
        max_new_tokens: 最大生成token数
        top_k: Top-k采样参数
        
    生成：
        SSE格式的JSON数据流，每个事件包含token或完成标志
    """
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # 累积token以正确处理多字节UTF-8字符（如emoji、中文）
    accumulated_tokens = []
    # 跟踪最后一次完整的UTF-8字符串（不含替换字符）
    last_clean_text = ""

    with worker.autocast_ctx:
        for token_column, token_masks in worker.engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=random.randint(0, 2**31 - 1)  # 随机种子确保多样性
        ):
            token = token_column[0]

            # 停止条件：遇到结束token
            if token == assistant_end or token == bos:
                break

            # 添加token到累积序列
            accumulated_tokens.append(token)
            # 解码所有累积的token以正确处理UTF-8
            # 注意：decode操作很高效，基本上就是查表和字符串拼接
            current_text = worker.tokenizer.decode(accumulated_tokens)
            # 只有当文本不以替换字符结尾时才发送
            # 这确保不会发送不完整的UTF-8序列
            if not current_text.endswith('�'):
                # 提取自上次清晰解码以来的新文本
                new_text = current_text[len(last_clean_text):]
                if new_text:  # 只有有新内容时才yield
                    yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                    last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    POST /chat/completions - 聊天补全API（仅支持流式响应）
    
    使用Worker池实现多GPU负载均衡
    
    请求格式：
    {
        "messages": [{"role": "user", "content": "你好"}],
        "temperature": 0.8,  // 可选
        "max_tokens": 512,   // 可选
        "top_k": 50          // 可选
    }
    
    响应格式（Server-Sent Events流）：
    data: {"token": "你", "gpu": 0}
    data: {"token": "好", "gpu": 0}
    ...
    data: {"done": true}
    """

    # 基本验证以防止滥用
    validate_chat_request(request)

    # 记录接收到的对话到控制台
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    # 从池中获取一个Worker（如果所有Worker都在忙则等待）
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # 构建对话token序列
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        # ============= 构建对话token序列 =============
        # 将对话历史转换为token序列，包含特殊标记
        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        # 添加助手开始标记，准备生成回复
        conversation_tokens.append(assistant_start)

        # ============= 流式响应和Worker管理 =============
        response_tokens = []
        
        async def stream_and_release():
            """
            流式生成助手回复并在完成后释放Worker
            
            关键设计：
            - 使用async generator实现流式传输
            - 累积响应token用于日志记录
            - finally块确保Worker一定会被释放（无论成功还是异常）
            """
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # 累积响应用于日志记录
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # 记录完整的助手回复到控制台
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # 流式传输完成后释放Worker回池中
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"  # Server-Sent Events格式
        )
    except Exception as e:
        # 确保即使发生错误也要释放Worker
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """
    GET /health - 健康检查端点
    
    返回服务器状态和Worker池信息，用于监控和负载均衡
    
    返回格式：
    {
        "status": "ok",
        "ready": true,
        "num_gpus": 4,
        "available_workers": 3
    }
    """
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """
    GET /stats - Worker池统计信息
    
    返回详细的Worker状态，包括每个Worker的GPU ID和设备信息
    
    返回格式：
    {
        "total_workers": 4,
        "available_workers": 3,
        "busy_workers": 1,
        "workers": [
            {"gpu_id": 0, "device": "cuda:0"},
            {"gpu_id": 1, "device": "cuda:1"},
            ...
        ]
    }
    """
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }

# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Web Server")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    # 启动Uvicorn ASGI服务器
    uvicorn.run(app, host=args.host, port=args.port)
