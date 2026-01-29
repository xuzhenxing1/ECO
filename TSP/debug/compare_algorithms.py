"""
对比最近邻算法和LKH求解器的解质量

可视化展示两种算法在相同问题上的表现差异
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from config import Config
from tsp_env import TSPEnv
from heuristics import get_nearest_neighbor_tour

try:
    from lkh_solver import LKHSolver
    LKH_AVAILABLE = True
except:
    LKH_AVAILABLE = False
    print("警告: LKH不可用，将只显示最近邻算法的结果")


def visualize_tour(coords, tour, title="TSP Tour", ax=None):
    """可视化TSP路径"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords
    tour_np = tour.cpu().numpy() if isinstance(tour, torch.Tensor) else tour
    
    # 绘制城市点
    ax.scatter(coords_np[:, 0], coords_np[:, 1], c='red', s=100, zorder=3)
    
    # 绘制路径
    for i in range(len(tour_np)):
        j = (i + 1) % len(tour_np)
        x1, y1 = coords_np[tour_np[i]]
        x2, y2 = coords_np[tour_np[j]]
        ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.6, linewidth=2)
    
    # 标记起点
    start_x, start_y = coords_np[tour_np[0]]
    ax.scatter([start_x], [start_y], c='green', s=200, marker='*', zorder=4, label='起点')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def compare_algorithms_visual(tsp_size=20, num_examples=3, save_dir="comparisons"):
    """可视化对比两种算法"""
    print(f"\n{'='*60}")
    print(f"可视化对比: 最近邻 vs LKH")
    print(f"{'='*60}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TSPEnv(device)
    
    if LKH_AVAILABLE:
        solver = LKHSolver()
    
    os.makedirs(save_dir, exist_ok=True)
    
    for example_idx in range(num_examples):
        print(f"生成示例 {example_idx + 1}/{num_examples}...")
        
        # 生成问题
        problem = env.get_random_problems(1, tsp_size).squeeze(0)
        
        # 最近邻求解
        with torch.no_grad():
            nn_tour = get_nearest_neighbor_tour(problem.unsqueeze(0)).squeeze(0)
        nn_length = env.get_tour_length(problem.unsqueeze(0), nn_tour.unsqueeze(0)).item()
        
        if LKH_AVAILABLE:
            # LKH求解
            lkh_tour_list = solver.solve(problem)
            lkh_tour = torch.tensor(lkh_tour_list, device=device)
            lkh_length = env.get_tour_length(problem.unsqueeze(0), lkh_tour.unsqueeze(0)).item()
            
            improvement = ((nn_length - lkh_length) / nn_length) * 100
            
            # 绘制对比图
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            visualize_tour(
                problem, nn_tour, 
                f"最近邻算法\n长度: {nn_length:.4f}", 
                axes[0]
            )
            
            visualize_tour(
                problem, lkh_tour,
                f"LKH算法\n长度: {lkh_length:.4f}\n改进: {improvement:.2f}%",
                axes[1]
            )
            
            plt.suptitle(f"TSP-{tsp_size} 示例 {example_idx + 1}", fontsize=16, fontweight='bold')
        else:
            # 只绘制最近邻
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            visualize_tour(
                problem, nn_tour,
                f"最近邻算法\nTSP-{tsp_size}\n长度: {nn_length:.4f}",
                ax
            )
        
        plt.tight_layout()
        
        filepath = os.path.join(save_dir, f"comparison_tsp{tsp_size}_example{example_idx + 1}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  保存到: {filepath}")
        plt.close()
    
    print(f"\n✓ 可视化完成! 图片保存在 {save_dir}/ 目录\n")


def compare_algorithms_statistics(tsp_size=50, num_samples=100):
    """统计对比两种算法"""
    print(f"\n{'='*60}")
    print(f"统计对比: 最近邻 vs LKH (TSP-{tsp_size}, {num_samples}样本)")
    print(f"{'='*60}\n")
    
    if not LKH_AVAILABLE:
        print("错误: LKH不可用，无法进行对比")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TSPEnv(device)
    solver = LKHSolver()
    
    nn_lengths = []
    lkh_lengths = []
    
    print("生成数据并求解...")
    for i in tqdm(range(num_samples)):
        # 生成问题
        problem = env.get_random_problems(1, tsp_size).squeeze(0)
        
        # 最近邻求解
        with torch.no_grad():
            nn_tour = get_nearest_neighbor_tour(problem.unsqueeze(0)).squeeze(0)
        nn_length = env.get_tour_length(problem.unsqueeze(0), nn_tour.unsqueeze(0)).item()
        nn_lengths.append(nn_length)
        
        # LKH求解
        lkh_tour_list = solver.solve(problem, verbose=False)
        lkh_tour = torch.tensor(lkh_tour_list, device=device)
        lkh_length = env.get_tour_length(problem.unsqueeze(0), lkh_tour.unsqueeze(0)).item()
        lkh_lengths.append(lkh_length)
    
    nn_lengths = np.array(nn_lengths)
    lkh_lengths = np.array(lkh_lengths)
    improvements = ((nn_lengths - lkh_lengths) / nn_lengths) * 100
    
    # 统计结果
    print(f"\n结果统计:")
    print(f"{'='*60}")
    print(f"最近邻算法:")
    print(f"  平均长度: {nn_lengths.mean():.4f} ± {nn_lengths.std():.4f}")
    print(f"  最小长度: {nn_lengths.min():.4f}")
    print(f"  最大长度: {nn_lengths.max():.4f}")
    print(f"\nLKH算法:")
    print(f"  平均长度: {lkh_lengths.mean():.4f} ± {lkh_lengths.std():.4f}")
    print(f"  最小长度: {lkh_lengths.min():.4f}")
    print(f"  最大长度: {lkh_lengths.max():.4f}")
    print(f"\n改进率:")
    print(f"  平均改进: {improvements.mean():.2f}%")
    print(f"  最小改进: {improvements.min():.2f}%")
    print(f"  最大改进: {improvements.max():.2f}%")
    print(f"{'='*60}\n")
    
    # 绘制分布图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 长度分布
    axes[0].hist(nn_lengths, bins=30, alpha=0.5, label='最近邻', color='blue')
    axes[0].hist(lkh_lengths, bins=30, alpha=0.5, label='LKH', color='green')
    axes[0].set_xlabel('路径长度')
    axes[0].set_ylabel('频数')
    axes[0].set_title('路径长度分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 改进率分布
    axes[1].hist(improvements, bins=30, color='orange', alpha=0.7)
    axes[1].axvline(improvements.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {improvements.mean():.2f}%')
    axes[1].set_xlabel('改进率 (%)')
    axes[1].set_ylabel('频数')
    axes[1].set_title('LKH相对最近邻的改进率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 散点图
    axes[2].scatter(nn_lengths, lkh_lengths, alpha=0.5, s=20)
    min_val = min(nn_lengths.min(), lkh_lengths.min())
    max_val = max(nn_lengths.max(), lkh_lengths.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    axes[2].set_xlabel('最近邻长度')
    axes[2].set_ylabel('LKH长度')
    axes[2].set_title('逐样本对比')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    filepath = f"comparisons/statistics_tsp{tsp_size}_n{num_samples}.png"
    os.makedirs("comparisons", exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"统计图保存到: {filepath}\n")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='对比最近邻和LKH算法')
    parser.add_argument('--mode', type=str, default='visual', choices=['visual', 'stats', 'both'],
                        help='对比模式: visual(可视化), stats(统计), both(两者)')
    parser.add_argument('--tsp_size', type=int, default=20,
                        help='TSP问题规模')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='统计模式的样本数量')
    parser.add_argument('--num_examples', type=int, default=3,
                        help='可视化模式的示例数量')
    
    args = parser.parse_args()
    
    if args.mode in ['visual', 'both']:
        compare_algorithms_visual(args.tsp_size, args.num_examples)
    
    if args.mode in ['stats', 'both']:
        compare_algorithms_statistics(args.tsp_size, args.num_samples)
    
    print("✓ 对比完成！")


if __name__ == "__main__":
    main()
