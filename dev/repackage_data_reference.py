"""
数据集重新打包参考脚本 - FinewebEdu-100B 数据集分片处理

功能说明：
本脚本用于将 FinewebEdu-100B 数据集重新打包成分片(shards)格式，以优化训练时的数据加载性能。

处理步骤：
1. 数据分片：将数据集切分成多个小文件，每个分片约100MB（经过zstd压缩后）
2. Parquet格式：使用 Apache Parquet 列式存储格式，行组(row group)大小设置为1024
3. 数据打乱：对整个数据集进行随机打乱(shuffle)，提高训练效果
4. 压缩优化：使用 zstd 压缩算法，平衡压缩率和解压速度

技术优势：
- 上传到 HuggingFace 进行托管，便于分布式访问
- 支持流式加载(streaming)：无需一次性加载全部数据到内存
- 本地磁盘缓存：边训练边缓存，显著降低训练延迟
- 适合大规模数据集的高效处理

重要提示：
本文件仅作为数据准备流程的参考文档和历史记录，不会在项目运行时执行。
实际训练时使用的是已经处理好并上传到 HuggingFace 的数据集。
"""
import os  # 文件和目录操作
import time  # 时间测量，用于进度估算

from datasets import load_dataset  # HuggingFace datasets库，用于加载数据集
import pyarrow.parquet as pq  # Apache Arrow的Parquet支持
import pyarrow as pa  # Apache Arrow核心库，高性能数据处理

# ============ 加载源数据集 ============
# 配置参数：从 HuggingFace 加载 FinewebEdu 数据集
dataset_kwargs = {
    "path": "HuggingFaceFW/fineweb-edu",  # 数据集路径
    "split": "train",  # 使用训练集分割
    "name": "sample-100BT",  # 子集名称：约100B个GPT-2 tokens（~3字符/token => ~300B字符总量）
}
ds = load_dataset(**dataset_kwargs)

# ============ 数据打乱 ============
# 随机打乱数据集顺序，使用固定种子以保证可复现性
# 打乱可以避免数据的局部相关性，提高模型训练效果
ds = ds.shuffle(seed=42)
ndocs = len(ds)  # 获取文档总数
print(f"文档总数: {ndocs}")

# ============ 输出目录设置 ============
# 指定Parquet文件的输出目录
output_dir = "/home/ubuntu/.cache/nanochat/base_data"
os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建

# ============ 分片参数配置 ============
chars_per_shard = 250_000_000  # 每个分片的目标字符数：2.5亿字符（约100MB压缩后）
row_group_size = 1024  # 行组大小：使用2的幂次，便于后续分布式数据加载器处理（HF默认使用1000）

# 初始化分片相关变量
shard_docs = []  # 当前分片累积的文档列表
shard_index = 0  # 当前分片索引
shard_characters = 0  # 当前分片累积的字符数
total_docs_processed = 0  # 已处理的文档总数
total_time_spent = 0  # 累计耗时（秒）
t0 = time.time()  # 记录开始时间

# ============ 主循环：遍历所有文档并生成分片 ============
for doc in ds:
    text = doc['text']  # 提取文档文本
    shard_docs.append(text)  # 添加到当前分片
    shard_characters += len(text)  # 累计字符数
    
    # 检查是否满足分片条件
    collected_enough_chars = shard_characters >= chars_per_shard  # 字符数达到目标
    docs_multiple_of_row_group_size = len(shard_docs) % row_group_size == 0  # 文档数是行组大小的倍数
    
    # 当两个条件都满足时，写入当前分片
    # 这样可以确保每个分片约100MB且行组边界对齐
    if collected_enough_chars and docs_multiple_of_row_group_size:
        # 构造分片文件路径，使用5位数字编号（如 shard_00000.parquet）
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        
        # 将文档列表转换为 Arrow Table
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        
        # 写入Parquet文件，配置优化参数
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=row_group_size,  # 设置行组大小
            use_dictionary=False,  # 不使用字典编码（通常用于分类数据，文本数据不适用）
            compression="zstd",  # 压缩算法：zstd（可选：NONE, SNAPPY, GZIP, BROTLI, LZ4, ZSTD）
            compression_level=3,  # 压缩级别：3（平衡压缩率和速度）
            write_statistics=False,  # 不写入统计信息（文本数据不需要）
        )
        
        # ============ 进度跟踪和时间估算 ============
        t1 = time.time()
        dt = t1 - t0  # 当前分片的处理耗时
        t0 = t1  # 重置计时器
        
        # 更新统计信息
        total_docs_processed += len(shard_docs)
        total_time_spent += dt
        
        # 计算剩余时间
        remaining_docs = ndocs - total_docs_processed
        avg_time_per_doc = total_time_spent / total_docs_processed
        remaining_time = remaining_docs * avg_time_per_doc
        remaining_time_hours = remaining_time / 3600
        
        # 输出进度信息
        print(f"已写入 {shard_path}。文档数: {len(shard_docs)} | 字符数: {shard_characters} | 耗时: {dt:.2f}秒 | 剩余时间: {remaining_time_hours:.2f}小时")
        
        # 重置当前分片的累积变量
        shard_docs = []
        shard_characters = 0
        shard_index += 1

# ============ 数据上传到 HuggingFace（示例代码） ============
# 下面的函数演示了如何将处理好的数据上传到 HuggingFace Hub
def upload():
    """
    将本地数据集文件夹上传到 HuggingFace Hub
    
    前提条件：
    - 需要设置环境变量 HF_TOKEN，包含 HuggingFace 访问令牌
    - 需要安装 huggingface_hub 库
    
    上传后的数据集可以：
    - 公开分享给其他研究者
    - 支持流式加载，无需下载完整数据集
    - 自动处理版本控制
    """
    import os
    from huggingface_hub import HfApi
    token = os.getenv("HF_TOKEN")  # 从环境变量获取访问令牌
    api = HfApi(token=token)
    api.upload_large_folder(
        folder_path=output_dir,  # 本地数据目录
        repo_id="karpathy/fineweb-edu-100b-shuffle",  # 目标仓库ID
        repo_type="dataset",  # 仓库类型：数据集
    )
# upload()  # 取消注释以执行上传
