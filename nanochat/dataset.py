"""
基础预训练数据集管理

基础/预训练数据集是一组Parquet文件。
本文件包含以下工具：
- 遍历Parquet文件并从中生成文档
- 按需下载文件（如果磁盘上不存在）

数据集详情：
    数据集：FineWeb-Edu-100BT（约100B token）
    格式：Parquet文件（已打乱顺序）
    来源：HuggingFace托管
    分片数：1823个分片（shard_00000.parquet到shard_01822.parquet）
    
数据集准备：
    有关数据集如何准备的详细信息，请参阅`dev/repackage_data_reference.py`。
    该脚本展示了如何将原始数据集重新打包成优化的Parquet格式。
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# =============================================================================
# 当前预训练数据集的具体信息
# =============================================================================

# 数据在互联网上托管的URL，按需下载
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # 最后一个数据分片是shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet"  # 文件名格式
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# 供其他模块导入使用的实用函数
# =============================================================================

def list_parquet_files(data_dir=None):
    """
    列出数据目录中的所有Parquet文件
    
    参数：
        data_dir: 数据目录路径（可选，默认使用DATA_DIR）
    
    返回：
        parquet_paths: Parquet文件的完整路径列表（已排序）
    
    注意：
        - 排除.tmp临时文件
        - 按文件名排序以确保一致的顺序
    """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, start=0, step=1):
    """
    以批次形式遍历数据集（基于底层row_groups以提高效率）
    
    参数：
        split: "train"或"val"。最后一个parquet文件用作验证集。
        start: 起始行组索引（用于DDP，如start=rank）
        step: 步长（用于DDP，如step=world_size）
    
    生成：
        texts: 文本列表（来自一个row_group）
    
    数据分割：
        - train: 前1822个文件（shard_00000到shard_01821）
        - val: 最后1个文件（shard_01822）
    
    分布式支持：
        通过start/step参数在多个进程间分片数据
        例如：start=rank, step=world_size
    
    效率：
        每次生成一个row_group的文本（通常约1024行），
        避免一次性读取整个文件到内存。
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# =============================================================================
# 数据集下载功能
# =============================================================================

def download_single_file(index):
    """
    下载单个文件，带重试和指数退避
    
    参数：
        index: 文件索引（0到MAX_SHARD）
    
    返回：
        bool: 下载是否成功
    
    特性：
        - 如果文件已存在，跳过下载
        - 先写入临时文件，成功后再重命名（原子操作）
        - 最多5次重试，指数退避（2^attempt秒）
        - 失败时清理部分下载的文件
        - 使用流式下载以节省内存（1MB块）
    """
    # 构造本地文件路径，如果已存在则跳过
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # 构造远程URL
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # 带重试的下载
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # 先写入临时文件
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB块
                    if chunk:
                        f.write(chunk)
            # 将临时文件移动到最终位置
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # 清理任何部分文件
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # 指数退避重试：2^attempt秒
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False

# =============================================================================
# 命令行下载工具
# =============================================================================

if __name__ == "__main__":
    """
    命令行下载工具主程序
    
    功能：
        并行下载FineWeb-Edu-100BT数据集的Parquet分片文件
    
    使用方法：
        python -m nanochat.dataset              # 下载所有分片（1823个）
        python -m nanochat.dataset -n 10        # 下载前10个分片
        python -m nanochat.dataset -w 8         # 使用8个并行下载线程
        python -m nanochat.dataset -n 100 -w 16 # 下载前100个分片，16个线程
    
    参数说明：
        -n, --num-files: 要下载的分片数量（默认-1=全部）
        -w, --num-workers: 并行下载线程数（默认4）
    
    输出：
        - 下载进度（每个文件）
        - 重试信息（失败时）
        - 最终统计（成功/总数）
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="下载FineWeb-Edu 100BT数据集分片")
    parser.add_argument("-n", "--num-files", type=int, default=-1, 
                       help="要下载的分片数量（默认-1=全部1823个分片）")
    parser.add_argument("-w", "--num-workers", type=int, default=4, 
                       help="并行下载线程数（默认4）")
    args = parser.parse_args()

    # 计算要下载的分片ID列表
    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    
    # 打印下载计划
    print(f"正在使用 {args.num_workers} 个线程下载 {len(ids_to_download)} 个分片...")
    print(f"目标目录: {DATA_DIR}")
    print()
    
    # 使用多进程池并行下载
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # 统计并报告下载结果
    successful = sum(1 for success in results if success)
    print(f"完成！成功下载: {successful}/{len(ids_to_download)} 个分片到 {DATA_DIR}")
