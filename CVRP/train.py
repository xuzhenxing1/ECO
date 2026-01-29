# -*- coding: utf-8 -*-
"""
CVRP训练脚本 - Mamba + DPO
完整流程：
1. Phase 0: SFT (Supervised Fine-Tuning) - 学习基础CVRP规则
2. Phase 1-4: Iterative DPO - 通过偏好学习优化解质量
"""

import torch
import torch.optim as optim
import copy
from tqdm import tqdm
import os
from datetime import datetime

from config import Config
from cvrp_env import CVRPEnv
from model import CVRPModel
from data_sampler import PreferenceSampler
from dpo_loss import dpo_loss
from temperature_scheduler import create_temperature_scheduler
from heuristics import get_nearest_neighbor_tour


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_run_dir(kind: str, problem_size: int) -> str:
    """创建带时间戳的运行目录"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = ensure_dir(Config.result_dir)
    run_dir = os.path.join(base, f"{kind}_{ts}_cvrp{problem_size}")
    ensure_dir(run_dir)
    return run_dir


def save_model(model, path: str):
    """保存模型权重"""
    torch.save(model.state_dict(), path)
    print(f"    [Save] Model saved to: {path}")


# ================= Phase 0: Supervised Fine-Tuning (SFT) =================
def run_sft_phase(policy_model, env, run_dir: str, sft_data_path=None):
    """
    SFT阶段：让模型学习基础的CVRP规则
    - 如何返回depot补货
    - 如何遵守容量约束
    - 基本的路径构建逻辑
    """
    print("\n>>> Phase 0: Starting Supervised Fine-Tuning (SFT)...")
    print("    Goal: Learn basic CVRP constraints and valid routing patterns.")
    
    optimizer = optim.Adam(policy_model.parameters(), lr=Config.sft_lr)
    policy_model.train()
    
    # 检查是否从文件加载数据
    if sft_data_path is None:
        # 尝试自动检测
        sft_data_path = Config.get_latest_data_file(Config.problem_size)
        if sft_data_path:
            print(f"    [Auto-detect] Found data file: {sft_data_path}")
    
    use_pregenerated_data = sft_data_path is not None and sft_data_path != "None" and os.path.exists(sft_data_path)
    
    if use_pregenerated_data:
        print(f"    [Data] Loading pre-generated SFT data from: {sft_data_path}")
        data = torch.load(sft_data_path)
        all_depot_xy = data['depot_xy'].to(Config.device)
        all_node_xy = data['node_xy'].to(Config.device)
        all_node_demand = data['node_demand'].to(Config.device)
        all_tours = data['tours'].to(Config.device)
        num_samples = all_depot_xy.size(0)
        
        print(f"    [Data] Loaded {num_samples} samples")
        print(f"    [Data] Problem Size: {data.get('problem_size', 'N/A')}")
        print(f"    [Data] Algorithm: {data.get('algorithm', 'N/A')}")
        print(f"    [Data] Avg Length: {data.get('avg_length', 'N/A'):.4f}")
        
        num_batches_per_epoch = (num_samples + Config.sft_batch_size - 1) // Config.sft_batch_size
    else:
        print("    [Data] No pre-generated data found, will generate data online using heuristics.")
        num_batches_per_epoch = 100
    
    for epoch in range(Config.sft_epochs):
        loss_sum = 0
        pbar = tqdm(range(num_batches_per_epoch), desc=f"SFT Epoch {epoch+1}")
        
        if use_pregenerated_data:
            indices = torch.randperm(num_samples, device=Config.device)
        
        for batch_idx in pbar:
            if use_pregenerated_data:
                # 从预生成数据中采样
                start_idx = batch_idx * Config.sft_batch_size
                end_idx = min(start_idx + Config.sft_batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                depot_xy = all_depot_xy[batch_indices]
                node_xy = all_node_xy[batch_indices]
                node_demand = all_node_demand[batch_indices]
                target_tours = all_tours[batch_indices]
            else:
                # 在线生成数据
                depot_xy, node_xy, node_demand = env.get_random_problems(
                    Config.sft_batch_size, 
                    Config.problem_size
                )
                
                # 使用启发式算法生成teacher解
                with torch.no_grad():
                    target_tours = get_nearest_neighbor_tour(
                        depot_xy, node_xy, node_demand, Config.vehicle_capacity
                    )
            
            # 计算Loss (Behavior Cloning)
            _, sum_log_probs = policy_model(
                depot_xy, node_xy, node_demand,
                target_tours, 
                teacher_forcing=True
            )
            
            # Maximize log_prob => Minimize -log_prob
            loss = -sum_log_probs.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            pbar.set_postfix({"SFT Loss": f"{loss.item():.4f}"})
    
    print(">>> SFT Phase Completed. Model now knows basic CVRP rules.\n")
    
    # 保存SFT checkpoint
    save_model(policy_model, os.path.join(run_dir, "sft_cvrp_model.pth"))
    return policy_model


# ================= Phase 1-4: Iterative DPO Loop =================
def run_iterative_dpo(policy_model, env, run_dir: str):
    """
    迭代DPO训练：
    1. 采样多个候选解
    2. 根据路径长度构建偏好对
    3. 用DPO loss优化模型
    4. 定期更新reference model
    """
    print(f">>> Starting Iterative DPO for {Config.total_iterations} iterations...")

    # 固定验证集
    x_eval_depot = None
    x_eval_nodes = None
    x_eval_demands = None
    
    if getattr(Config, 'eval_use_fixed_set', False):
        n_eval = int(getattr(Config, 'eval_batch_size', 500))
        x_eval_depot, x_eval_nodes, x_eval_demands = env.get_random_problems(
            n_eval, Config.problem_size
        )
    
    # 加载离线高质量数据（用于混合训练）
    offline_depot = None
    offline_nodes = None
    offline_demands = None
    offline_tours = None
    
    if getattr(Config, 'use_hybrid_data', False):
        hybrid_data_path = getattr(Config, 'hybrid_data_path', None)
        if hybrid_data_path is None:
            hybrid_data_path = Config.sft_data_path
            # 如果sft_data_path也是None，尝试自动检测
            if hybrid_data_path is None:
                hybrid_data_path = Config.get_latest_data_file(Config.problem_size)
        
        if hybrid_data_path and hybrid_data_path != "None" and os.path.exists(hybrid_data_path):
            print(f"[Hybrid Data] Loading offline high-quality data from: {hybrid_data_path}")
            data = torch.load(hybrid_data_path)
            offline_depot = data['depot_xy'].to(Config.device)
            offline_nodes = data['node_xy'].to(Config.device)
            offline_demands = data['node_demand'].to(Config.device)
            offline_tours = data['tours'].to(Config.device)
            offline_ratio = getattr(Config, 'hybrid_offline_ratio', 0.3)
            print(f"[Hybrid Data] Loaded {offline_depot.size(0)} samples")
            print(f"[Hybrid Data] Offline ratio: {offline_ratio:.1%}")
        else:
            print(f"[Hybrid Data] Warning: use_hybrid_data=True but data file not found")
    
    # 初始化Reference Model
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    
    optimizer = optim.Adam(policy_model.parameters(), lr=Config.dpo_lr)
    sampler = PreferenceSampler(policy_model, env)
    
    # 创建温度调度器
    temp_scheduler = create_temperature_scheduler(Config)
    print(f"[Temperature] Strategy: {getattr(Config, 'temperature_decay', 'fixed')}")
    
    history = []
    best_eval_length = float("inf")
    best_model_path = os.path.join(run_dir, "best_cvrp_model.pth")

    # 外层循环：迭代轮次
    for iteration in range(Config.total_iterations):
        # 获取当前温度
        current_temp = temp_scheduler.get_temperature(iteration)
        temp_info = temp_scheduler.get_info(iteration) if hasattr(temp_scheduler, 'get_info') else {}
        
        print(f"\n--- Iteration {iteration+1} / {Config.total_iterations} ---")
        if temp_info:
            print(f"    Temperature: {current_temp:.3f} | Phase: {temp_info.get('phase', 'N/A')}")
        
        policy_model.train()
        iter_loss = 0
        
        # 内层循环：在当前数据分布上多次更新
        pbar = tqdm(range(Config.epochs_per_iter * 20), desc=f"Iter {iteration+1} Training")
        
        for step in pbar:
            # 混合数据策略
            use_hybrid = (offline_depot is not None and getattr(Config, 'use_hybrid_data', False))
            
            if use_hybrid:
                # 计算在线和离线数据的数量
                offline_ratio = getattr(Config, 'hybrid_offline_ratio', 0.3)
                n_offline = int(Config.dpo_batch_size * offline_ratio)
                n_online = Config.dpo_batch_size - n_offline
                
                # A1. 在线生成部分数据
                depot_online, nodes_online, demands_online = env.get_random_problems(
                    n_online, Config.problem_size
                )
                depot_online, nodes_online, demands_online, winner_online, loser_online = \
                    sampler.sample_dpo_data(depot_online, nodes_online, demands_online, temperature=current_temp)
                
                # A2. 从离线数据中随机采样
                indices = torch.randperm(offline_depot.size(0), device=Config.device)[:n_offline]
                depot_offline = offline_depot[indices]
                nodes_offline = offline_nodes[indices]
                demands_offline = offline_demands[indices]
                tours_offline = offline_tours[indices]
                
                # 对离线数据也进行采样
                depot_offline, nodes_offline, demands_offline, winner_offline, loser_offline = \
                    sampler.sample_dpo_data(
                        depot_offline, nodes_offline, demands_offline,
                        temperature=current_temp,
                        reference_tour=tours_offline
                    )
                
                # A3. 合并在线和离线数据
                depot_xy = torch.cat([depot_online, depot_offline], dim=0)
                node_xy = torch.cat([nodes_online, nodes_offline], dim=0)
                node_demand = torch.cat([demands_online, demands_offline], dim=0)
                winner_tours = torch.cat([winner_online, winner_offline], dim=0)
                loser_tours = torch.cat([loser_online, loser_offline], dim=0)
                
            else:
                # 纯在线生成
                depot_xy, node_xy, node_demand = env.get_random_problems(
                    Config.dpo_batch_size, Config.problem_size
                )
                
                depot_xy, node_xy, node_demand, winner_tours, loser_tours = \
                    sampler.sample_dpo_data(depot_xy, node_xy, node_demand, temperature=current_temp)
            
            # 跳过无效偏好对
            valid = (winner_tours != loser_tours).any(dim=1)
            if not valid.any():
                continue
            
            depot_xy = depot_xy[valid]
            node_xy = node_xy[valid]
            node_demand = node_demand[valid]
            winner_tours = winner_tours[valid]
            loser_tours = loser_tours[valid]
            
            # 计算Policy LogProbs
            _, policy_chosen_logps = policy_model(
                depot_xy, node_xy, node_demand,
                winner_tours, teacher_forcing=True
            )
            _, policy_rejected_logps = policy_model(
                depot_xy, node_xy, node_demand,
                loser_tours, teacher_forcing=True
            )
            
            # 计算Ref LogProbs (No Grad)
            with torch.no_grad():
                _, ref_chosen_logps = ref_model(
                    depot_xy, node_xy, node_demand,
                    winner_tours, teacher_forcing=True
                )
                _, ref_rejected_logps = ref_model(
                    depot_xy, node_xy, node_demand,
                    loser_tours, teacher_forcing=True
                )
            
            # Per-step归一化
            if getattr(Config, 'normalize_logp_by_tour_len', False):
                # 使用实际tour长度归一化
                # 简化：使用平均tour长度
                avg_tour_len = (winner_tours != 0).sum(dim=1).float().mean()
                denom = avg_tour_len.item() if avg_tour_len > 0 else Config.problem_size
                
                policy_chosen_logps = policy_chosen_logps / denom
                policy_rejected_logps = policy_rejected_logps / denom
                ref_chosen_logps = ref_chosen_logps / denom
                ref_rejected_logps = ref_rejected_logps / denom
            
            # 计算DPO Loss
            loss, loss_val = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=Config.dpo_beta
            )
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            
            iter_loss += loss_val
            pbar.set_postfix({"DPO Loss": f"{loss_val:.4f}"})
        
        avg_loss = iter_loss / (Config.epochs_per_iter * 20)
        
        # 评估
        if x_eval_depot is not None:
            policy_model.eval()
            with torch.no_grad():
                eval_tours, _ = policy_model(
                    x_eval_depot, x_eval_nodes, x_eval_demands,
                    teacher_forcing=False, temperature=1.0
                )
                eval_lengths = env.get_tour_length(x_eval_depot, x_eval_nodes, eval_tours)
                avg_eval_length = eval_lengths.mean().item()
            
            print(f"    Avg Loss: {avg_loss:.4f} | Eval Length: {avg_eval_length:.4f}")
            
            # 保存最佳模型
            if avg_eval_length < best_eval_length:
                best_eval_length = avg_eval_length
                save_model(policy_model, best_model_path)
                print(f"    ✓ New best model! Length: {best_eval_length:.4f}")
            
            history.append({
                'iteration': iteration + 1,
                'loss': avg_loss,
                'eval_length': avg_eval_length,
                'temperature': current_temp
            })
        
        # 更新Reference Model
        if Config.update_ref_model and (iteration + 1) % Config.ref_update_interval == 0:
            print(f"    [Update] Syncing reference model...")
            ref_model.load_state_dict(policy_model.state_dict())
            ref_model.eval()
        
        # 定期保存checkpoint
        if (iteration + 1) % 10 == 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_iter{iteration+1}.pth")
            save_model(policy_model, checkpoint_path)
    
    print("\n>>> Iterative DPO Completed!")
    print(f"    Best Eval Length: {best_eval_length:.4f}")
    
    return policy_model, history


# ================= 主函数 =================
def main():
    print("="*60)
    print("CVRP Training: Mamba + DPO")
    print("="*60)
    print(f"Problem Size: {Config.problem_size}")
    print(f"Device: {Config.device}")
    print(f"Embedding Dim: {Config.embedding_dim}")
    print(f"Encode Layers: {Config.n_encode_layers}")
    print("="*60 + "\n")
    
    # 创建运行目录
    run_dir = make_run_dir("train", Config.problem_size)
    print(f"Run directory: {run_dir}\n")
    
    # 初始化环境
    env = CVRPEnv(Config.device)
    
    # 初始化模型
    policy_model = CVRPModel(
        embedding_dim=Config.embedding_dim,
        hidden_dim=Config.embedding_dim,
        n_encode_layers=Config.n_encode_layers
    ).to(Config.device)
    
    print(f"Model parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    
    # Phase 0: SFT
    policy_model = run_sft_phase(policy_model, env, run_dir, Config.sft_data_path)
    
    # Phase 1-4: Iterative DPO
    policy_model, history = run_iterative_dpo(policy_model, env, run_dir)
    
    # 保存训练历史
    torch.save(history, os.path.join(run_dir, "training_history.pt"))
    
    print(f"\n✓ Training completed! Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
