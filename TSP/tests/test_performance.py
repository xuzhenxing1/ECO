# -*- coding: utf-8 -*-
"""
性能优化测试脚本

对比不同优化策略的效果：
1. 基线：标准评估（单次 greedy）
2. POMO 评估（多起点搜索）
3. 温度采样
4. 使用 LKH 数据训练的模型
"""

import torch
from config import Config
from model import AttentionModel
from tsp_env import TSPEnv
import time
import os

def test_evaluation_strategies(model, env, n_problems=100):
    """测试不同评估策略"""
    print("\n" + "="*60)
    print("测试不同评估策略")
    print("="*60)
    
    # 生成测试问题
    x = env.get_random_problems(n_problems, Config.tsp_size)
    model.eval()
    
    # 1. 标准 Greedy 评估
    print("\n[1] 标准 Greedy 评估（单次）")
    start = time.time()
    with torch.no_grad():
        tours, _ = model(x, teacher_forcing=False)
        lengths = env.get_tour_length(x, tours)
    greedy_len = lengths.mean().item()
    greedy_time = time.time() - start
    print(f"    平均长度: {greedy_len:.4f}")
    print(f"    耗时: {greedy_time:.2f}s")
    
    # 2. POMO 评估（多起点）
    print("\n[2] POMO 评估（多起点搜索）")
    for pomo_size in [10, 20, Config.tsp_size]:
        start = time.time()
        pomo_len = evaluate_pomo_simple(model, env, x, pomo_size)
        pomo_time = time.time() - start
        improvement = (greedy_len - pomo_len) / greedy_len * 100
        print(f"    POMO-{pomo_size:2d}: {pomo_len:.4f} (改进 {improvement:+.2f}%, 耗时 {pomo_time:.2f}s)")
    
    # 3. 采样 + 选最优
    print("\n[3] 多次采样 + 选最优")
    model.train()  # 启用采样模式
    for num_samples in [10, 20, 50]:
        start = time.time()
        sample_len = evaluate_sampling(model, env, x, num_samples)
        sample_time = time.time() - start
        improvement = (greedy_len - sample_len) / greedy_len * 100
        print(f"    Sampling-{num_samples:2d}: {sample_len:.4f} (改进 {improvement:+.2f}%, 耗时 {sample_time:.2f}s)")
    model.eval()
    
    # 4. 不同温度的采样
    print("\n[4] 不同温度采样")
    model.train()
    for temp in [0.5, 0.8, 1.0, 1.2, 1.5]:
        start = time.time()
        temp_len = evaluate_sampling(model, env, x, 20, temperature=temp)
        temp_time = time.time() - start
        improvement = (greedy_len - temp_len) / greedy_len * 100
        print(f"    Temp={temp:.1f}: {temp_len:.4f} (改进 {improvement:+.2f}%, 耗时 {temp_time:.2f}s)")
    model.eval()


def evaluate_pomo_simple(model, env, x, pomo_size):
    """简化的 POMO 评估（不旋转路径，只是多次采样）"""
    B, N, _ = x.size()
    all_lengths = []
    
    with torch.no_grad():
        for _ in range(pomo_size):
            tours, _ = model(x, teacher_forcing=False)
            lengths = env.get_tour_length(x, tours)
            all_lengths.append(lengths)
    
    all_lengths = torch.stack(all_lengths, dim=1)
    best_lengths = all_lengths.min(dim=1)[0]
    return best_lengths.mean().item()


def evaluate_sampling(model, env, x, num_samples, temperature=1.0):
    """采样评估"""
    B, N, _ = x.size()
    all_lengths = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            tours, _ = model(x, teacher_forcing=False, temperature=temperature)
            lengths = env.get_tour_length(x, tours)
            all_lengths.append(lengths)
    
    all_lengths = torch.stack(all_lengths, dim=1)
    best_lengths = all_lengths.min(dim=1)[0]
    return best_lengths.mean().item()


def compare_with_lkh(model, env, n_problems=50):
    """与 LKH 对比"""
    print("\n" + "="*60)
    print("与 LKH 最优解对比")
    print("="*60)
    
    try:
        from lkh_solver import LKHSolver
        solver = LKHSolver(lkh_path=Config.lkh_path)
        
        # 生成测试问题
        x = env.get_random_problems(n_problems, Config.tsp_size)
        
        # 神经网络解（POMO）
        print("\n[Neural Network + POMO]")
        nn_len = evaluate_pomo_simple(model, env, x, pomo_size=20)
        print(f"    平均长度: {nn_len:.4f}")
        
        # LKH 解
        print("\n[LKH 求解器]")
        lkh_lengths = []
        x_np = x.cpu().numpy()
        for i in range(n_problems):
            coords = x_np[i]
            tour = solver.solve(coords, runs=3, max_trials=500)
            # 计算长度
            tour_tensor = torch.tensor(tour, device=x.device).unsqueeze(0)
            length = env.get_tour_length(x[i:i+1], tour_tensor)
            lkh_lengths.append(length.item())
        
        lkh_len = sum(lkh_lengths) / len(lkh_lengths)
        print(f"    平均长度: {lkh_len:.4f}")
        
        # Gap
        gap = (nn_len - lkh_len) / lkh_len * 100
        print(f"\n[Gap] {gap:.2f}% (越小越好)")
        
        if gap < 5:
            print("    ✓ 优秀！接近最优解")
        elif gap < 10:
            print("    ○ 良好，还有提升空间")
        elif gap < 20:
            print("    △ 一般，建议优化")
        else:
            print("    ✗ 较差，需要改进")
            
    except Exception as e:
        print(f"\n    无法运行 LKH 对比: {e}")


def main():
    print("="*60)
    print("TSP-DPO 性能优化测试")
    print("="*60)
    
    # 初始化
    env = TSPEnv(Config.device)
    
    # 查找最新的模型
    result_dirs = [d for d in os.listdir('result') if d.startswith('train_')]
    if not result_dirs:
        print("\n错误：找不到训练好的模型")
        print("请先运行 train.py 进行训练")
        return
    
    latest_dir = max(result_dirs)
    best_model_path = os.path.join('result', latest_dir, 'best_tsp_model.pth')
    
    if not os.path.exists(best_model_path):
        print(f"\n错误：找不到模型文件 {best_model_path}")
        return
    
    print(f"\n加载模型: {best_model_path}")
    
    # 初始化模型
    embedding_dim = getattr(Config, 'embedding_dim', 128)
    n_encode_layers = getattr(Config, 'n_encode_layers', 3)
    
    model = AttentionModel(
        embedding_dim=embedding_dim,
        hidden_dim=embedding_dim,
        n_heads=8,
        n_encode_layers=n_encode_layers
    ).to(Config.device)
    
    # 加载权重
    checkpoint = torch.load(best_model_path, map_location=Config.device)
    model.load_state_dict(checkpoint)
    
    print(f"    模型配置: embed_dim={embedding_dim}, n_layers={n_encode_layers}")
    print(f"    TSP 规模: {Config.tsp_size} 城市")
    
    # 运行测试
    test_evaluation_strategies(model, env, n_problems=100)
    compare_with_lkh(model, env, n_problems=20)
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print("\n💡 优化建议：")
    print("1. 使用 POMO 评估可显著提升解质量（建议设置 use_pomo_eval=True）")
    print("2. 使用 LKH 生成的高质量 SFT 数据进行训练")
    print("3. 适当增加模型容量（embedding_dim=256, n_layers=6）")
    print("4. 增加 DPO 采样数量（num_samples=32）")
    print("5. 调整采样温度平衡探索/利用（temperature=1.0-1.2）")


if __name__ == "__main__":
    main()
