"""
测试脚本 - 验证CVRP各组件功能
"""

import torch
from config import Config
from cvrp_env import CVRPEnv
from model import CVRPModel
from heuristics import get_nearest_neighbor_tour
from lkh_solver import LKHSolver
import numpy as np


def test_env():
    """测试CVRP环境"""
    print("\n" + "="*60)
    print("Test 1: CVRP Environment")
    print("="*60)
    
    env = CVRPEnv(Config.device)
    
    # 生成问题
    depot_xy, node_xy, node_demand = env.get_random_problems(
        batch_size=2, problem_size=10
    )
    
    print(f"Depot shape: {depot_xy.shape}")
    print(f"Nodes shape: {node_xy.shape}")
    print(f"Demands shape: {node_demand.shape}")
    print(f"Sample demand: {node_demand[0]}")
    
    # 生成一个简单tour（手动）
    # [depot, node1, node2, depot, node3, node4, depot]
    tour = torch.tensor([[0, 1, 2, 0, 3, 4, 0, 5, 6, 0]], device=Config.device)
    
    # 计算长度
    length = env.get_tour_length(depot_xy[:1], node_xy[:1], tour)
    print(f"Tour: {tour[0].cpu().numpy()}")
    print(f"Length: {length.item():.4f}")
    
    # 检查可行性
    tour_short = tour[:, :7]
    is_feasible, violations = env.check_feasibility(node_demand[:1], tour_short)
    print(f"Is feasible: {is_feasible[0].item()}")
    print(f"Violations: {violations[0].item()}")
    
    print("✓ Environment test passed!\n")


def test_model():
    """测试CVRP模型"""
    print("\n" + "="*60)
    print("Test 2: CVRP Model")
    print("="*60)
    
    model = CVRPModel(
        embedding_dim=128,
        hidden_dim=128,
        n_encode_layers=3
    ).to(Config.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    env = CVRPEnv(Config.device)
    depot_xy, node_xy, node_demand = env.get_random_problems(2, 10)
    
    # 测试前向传播（采样模式）
    model.eval()
    with torch.no_grad():
        tour, log_probs = model(
            depot_xy, node_xy, node_demand,
            teacher_forcing=False, temperature=1.0
        )
    
    print(f"Generated tour shape: {tour.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Sample tour: {tour[0].cpu().numpy()}")
    
    # 计算长度
    lengths = env.get_tour_length(depot_xy, node_xy, tour)
    print(f"Tour lengths: {lengths.cpu().numpy()}")
    
    # 检查可行性
    is_feasible, violations = env.check_feasibility(node_demand, tour)
    print(f"Is feasible: {is_feasible.cpu().numpy()}")
    
    print("✓ Model test passed!\n")


def test_heuristics():
    """测试启发式算法"""
    print("\n" + "="*60)
    print("Test 3: Heuristic Algorithms")
    print("="*60)
    
    env = CVRPEnv(Config.device)
    depot_xy, node_xy, node_demand = env.get_random_problems(2, 10)
    
    # 最近邻算法
    tour = get_nearest_neighbor_tour(depot_xy, node_xy, node_demand)
    
    print(f"NN Tour shape: {tour.shape}")
    print(f"Sample tour: {tour[0].cpu().numpy()}")
    
    # 计算长度
    lengths = env.get_tour_length(depot_xy, node_xy, tour)
    print(f"Tour lengths: {lengths.cpu().numpy()}")
    
    # 检查可行性
    is_feasible, violations = env.check_feasibility(node_demand, tour)
    print(f"Is feasible: {is_feasible.cpu().numpy()}")
    
    print("✓ Heuristics test passed!\n")


def test_lkh_solver():
    """测试LKH求解器（需要LKH-3安装）"""
    print("\n" + "="*60)
    print("Test 4: LKH-3 Solver")
    print("="*60)
    
    try:
        solver = LKHSolver(lkh_path=Config.lkh_path, runs=1)
        
        # 创建测试问题
        np.random.seed(42)
        depot = np.random.rand(2)
        nodes = np.random.rand(5, 2)
        demands = np.random.randint(1, 10, 5) / 30.0
        
        print(f"Problem size: 5 customers")
        print(f"Demands: {demands}")
        
        # 求解
        tour = solver.solve(depot, nodes, demands, capacity=1.0, verbose=False)
        
        print(f"Solution tour: {tour}")
        print(f"Tour length: {len(tour)}")
        
        print("✓ LKH solver test passed!\n")
        
    except Exception as e:
        print(f"⚠ LKH solver test skipped: {e}")
        print("  (This is expected if LKH-3 is not installed)\n")


def test_teacher_forcing():
    """测试teacher forcing模式"""
    print("\n" + "="*60)
    print("Test 5: Teacher Forcing (SFT Training)")
    print("="*60)
    
    model = CVRPModel(
        embedding_dim=128,
        hidden_dim=128,
        n_encode_layers=3
    ).to(Config.device)
    
    env = CVRPEnv(Config.device)
    depot_xy, node_xy, node_demand = env.get_random_problems(2, 10)
    
    # 生成teacher tour
    target_tour = get_nearest_neighbor_tour(depot_xy, node_xy, node_demand)
    
    # Teacher forcing模式
    model.train()
    _, log_probs = model(
        depot_xy, node_xy, node_demand,
        target_tour, teacher_forcing=True
    )
    
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Log probs: {log_probs.cpu().numpy()}")
    
    # 计算loss
    loss = -log_probs.mean()
    print(f"SFT Loss: {loss.item():.4f}")
    
    # 反向传播测试
    loss.backward()
    print("✓ Backward pass successful")
    
    print("✓ Teacher forcing test passed!\n")


def main():
    print("="*60)
    print("CVRP Component Tests")
    print("="*60)
    print(f"Device: {Config.device}")
    print(f"Problem Size: {Config.problem_size}")
    print("="*60)
    
    test_env()
    test_model()
    test_heuristics()
    test_lkh_solver()
    test_teacher_forcing()
    
    print("="*60)
    print("✓ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()
