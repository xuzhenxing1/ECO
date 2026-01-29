"""
生成CVRP的SFT训练数据
使用LKH-3求解器生成高质量解
"""

import torch
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from config import Config
from cvrp_env import CVRPEnv
from lkh_solver import LKHSolver


def generate_sft_dataset(num_samples=100, problem_size=10, save_dir=None):
    """
    生成SFT训练数据集
    
    Args:
        num_samples: 生成样本数量
        problem_size: 问题规模（客户节点数）
        save_dir: 保存目录，默认使用Config.data_dir
    """
    if save_dir is None:
        save_dir = Config.data_dir
    
    print("="*60)
    print(f"Generating SFT Dataset for CVRP-{problem_size}")
    print("="*60)
    print(f"Number of samples: {num_samples}")
    print(f"Problem size: {problem_size}")
    print(f"Save directory: {save_dir}")
    print(f"Using LKH-3 solver")
    print("="*60 + "\n")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化环境和求解器（使用CPU避免CUDA错误）
    env = CVRPEnv(device='cpu')
    solver = LKHSolver(
        lkh_path=Config.lkh_path,
        runs=Config.lkh_runs,
        max_trials=Config.lkh_max_trials
    )
    
    all_depot_xy = []
    all_node_xy = []
    all_node_demand = []
    all_tours = []
    all_lengths = []
    
    # 批量生成
    batch_size = 10  # 每批生成10个样本
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating data"):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # 生成问题
        depot_xy, node_xy, node_demand = env.get_random_problems(
            current_batch_size, problem_size
        )
        
        # 使用LKH求解每个问题
        for i in range(current_batch_size):
            # depot_xy: [Batch, 1, 2] -> 取[i, 0, :]得到单个depot的坐标
            depot = depot_xy[i, 0, :].cpu().numpy()  # shape: (2,)
            nodes = node_xy[i].cpu().numpy()  # shape: (problem_size, 2)
            demands = node_demand[i].cpu().numpy()  # shape: (problem_size,)
            
            try:
                # LKH求解
                tour = solver.solve(depot, nodes, demands, Config.vehicle_capacity)
                
                # 计算路径长度（在CPU上）
                tour_tensor = torch.tensor(tour, dtype=torch.long, device='cpu')
                length = env.get_tour_length(
                    depot_xy[i:i+1], 
                    node_xy[i:i+1], 
                    tour_tensor.unsqueeze(0)
                ).item()
                
                # 保存（移到目标设备）
                all_depot_xy.append(depot_xy[i].to(Config.device))
                all_node_xy.append(node_xy[i].to(Config.device))
                all_node_demand.append(node_demand[i].to(Config.device))
                all_tours.append(tour_tensor.to(Config.device))
                all_lengths.append(length)
                
            except Exception as e:
                print(f"\nWarning: Failed to solve sample {batch_idx * batch_size + i}: {e}")
                continue
    
    # 转换为tensor
    # 注意：tours可能长度不同，需要padding
    max_tour_len = max(len(t) for t in all_tours)
    
    padded_tours = []
    for tour in all_tours:
        if len(tour) < max_tour_len:
            padding = torch.zeros(max_tour_len - len(tour), dtype=torch.long, device=tour.device)
            tour = torch.cat([tour, padding])
        padded_tours.append(tour)
    
    dataset = {
        'depot_xy': torch.stack(all_depot_xy),
        'node_xy': torch.stack(all_node_xy),
        'node_demand': torch.stack(all_node_demand),
        'tours': torch.stack(padded_tours),
        'lengths': torch.tensor(all_lengths),
        'problem_size': problem_size,
        'num_samples': len(all_tours),
        'avg_length': np.mean(all_lengths),
        'std_length': np.std(all_lengths),
        'algorithm': 'LKH-3',
        'lkh_runs': Config.lkh_runs,
        'lkh_max_trials': Config.lkh_max_trials
    }
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sft_data_lkh_cvrp{problem_size}_n{len(all_tours)}_{timestamp}.pt"
    filepath = os.path.join(save_dir, filename)
    
    torch.save(dataset, filepath)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"  Saved to: {filepath}")
    print(f"  Total samples: {len(all_tours)}")
    print(f"  Average length: {dataset['avg_length']:.4f}")
    print(f"  Std length: {dataset['std_length']:.4f}")
    
    return filepath


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CVRP SFT Dataset")
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--problem_size', type=int, default=10, help='Problem size (number of customers)')
    parser.add_argument('--save_dir', type=str, default='data', help='Save directory')
    
    args = parser.parse_args()
    
    generate_sft_dataset(
        num_samples=args.num_samples,
        problem_size=args.problem_size,
        save_dir=args.save_dir
    )
