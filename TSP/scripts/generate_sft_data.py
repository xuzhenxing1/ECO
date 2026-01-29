"""
独立的SFT数据生成脚本
使用最近邻（Nearest Neighbor）启发式算法生成TSP训练数据
可以单独运行，生成的数据保存到 data/ 文件夹

Usage:
    python generate_sft_data.py --tsp_size 100 --num_samples 10000
"""

import torch
import argparse
import os
from datetime import datetime
from tqdm import tqdm

from config import Config
from tsp_env import TSPEnv
from heuristics import get_nearest_neighbor_tour


def ensure_dir(path: str) -> str:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def generate_sft_data(tsp_size, num_samples, batch_size=256, save_dir="data"):
    """
    生成SFT训练数据
    
    Args:
        tsp_size: TSP问题规模（城市数量）
        num_samples: 要生成的样本总数
        batch_size: 批处理大小
        save_dir: 数据保存目录
    
    Returns:
        保存的文件路径
    """
    print(f"\n{'='*60}")
    print(f"生成SFT数据")
    print(f"{'='*60}")
    print(f"TSP规模: {tsp_size}")
    print(f"样本数量: {num_samples}")
    print(f"批处理大小: {batch_size}")
    print(f"保存目录: {save_dir}")
    print(f"{'='*60}\n")
    
    # 初始化环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    env = TSPEnv(device)
    
    # 创建保存目录
    ensure_dir(save_dir)
    
    # 准备存储数据的列表
    all_problems = []
    all_tours = []
    
    # 分批生成数据
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print("开始生成数据...")
    for batch_idx in tqdm(range(num_batches), desc="生成进度"):
        # 计算当前批次的实际大小
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # 1. 生成随机TSP问题
        problems = env.get_random_problems(current_batch_size, tsp_size)
        
        # 2. 使用最近邻算法生成解（作为SFT的标签）
        with torch.no_grad():
            tours = get_nearest_neighbor_tour(problems)
        
        # 3. 将数据移到CPU并存储
        all_problems.append(problems.cpu())
        all_tours.append(tours.cpu())
    
    # 合并所有批次的数据
    all_problems = torch.cat(all_problems, dim=0)
    all_tours = torch.cat(all_tours, dim=0)
    
    # 只保留需要的样本数量（处理最后一个batch可能多的情况）
    all_problems = all_problems[:num_samples]
    all_tours = all_tours[:num_samples]
    
    print(f"\n数据生成完成！")
    print(f"问题形状: {all_problems.shape}")  # [num_samples, tsp_size, 2]
    print(f"路径形状: {all_tours.shape}")      # [num_samples, tsp_size]
    
    # 计算一些统计信息
    with torch.no_grad():
        problems_gpu = all_problems.to(device)
        tours_gpu = all_tours.to(device)
        lengths = env.get_tour_length(problems_gpu, tours_gpu)
        avg_length = lengths.mean().item()
        std_length = lengths.std().item()
        min_length = lengths.min().item()
        max_length = lengths.max().item()
    
    print(f"\n生成数据的统计信息:")
    print(f"  平均路径长度: {avg_length:.4f}")
    print(f"  标准差: {std_length:.4f}")
    print(f"  最小路径长度: {min_length:.4f}")
    print(f"  最大路径长度: {max_length:.4f}")
    
    # 保存数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sft_data_tsp{tsp_size}_n{num_samples}_{timestamp}.pt"
    filepath = os.path.join(save_dir, filename)
    
    torch.save({
        'problems': all_problems,      # [num_samples, tsp_size, 2]
        'tours': all_tours,            # [num_samples, tsp_size]
        'tsp_size': tsp_size,
        'num_samples': num_samples,
        'avg_length': avg_length,
        'std_length': std_length,
        'min_length': min_length,
        'max_length': max_length,
        'timestamp': timestamp,
        'algorithm': 'nearest_neighbor'
    }, filepath)
    
    print(f"\n数据已保存到: {filepath}")
    print(f"{'='*60}\n")
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description='生成TSP的SFT训练数据')
    parser.add_argument('--tsp_size', type=int, default=None,
                        help='TSP问题规模（城市数量），默认使用config.py中的值')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='生成的样本数量（默认: 10000）')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批处理大小（默认: 256）')
    parser.add_argument('--save_dir', type=str, default='data',
                        help='数据保存目录（默认: data）')
    
    args = parser.parse_args()
    
    # 如果没有指定tsp_size，使用config中的值
    tsp_size = args.tsp_size if args.tsp_size is not None else Config.tsp_size
    
    # 生成数据
    filepath = generate_sft_data(
        tsp_size=tsp_size,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    
    print("✓ 数据生成成功！")
    print(f"可以在训练时使用: --sft_data_path {filepath}")


if __name__ == "__main__":
    main()
