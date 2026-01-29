# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from config import Config
from tsp_env import TSPEnv
from model import AttentionModel
from data_sampler import PreferenceSampler
from dpo_loss import dpo_loss
from temperature_scheduler import create_temperature_scheduler
from heuristics import get_nearest_neighbor_tour


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_run_dir(kind: str, tsp_size: int) -> str:
    """Create a timestamped run directory under ./results."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = ensure_dir("result")
    run_dir = os.path.join(base, f"{kind}_{ts}_tsp{tsp_size}")
    ensure_dir(run_dir)
    return run_dir


def save_model(model, path: str):
    """保存模型权重（state_dict）"""
    torch.save(model.state_dict(), path)
    print(f"    [Save] Model saved to: {path}")

# ================= Phase 0: Supervised Fine-Tuning (SFT) =================
def run_sft_phase(policy_model, env, run_dir: str, sft_data_path=None):
    print("\n>>> Phase 0: Starting Supervised Fine-Tuning (SFT)...")
    print("    Goal: Clone behavior of a Greedy Heuristic to learn valid TSP constraints.")
    
    optimizer = optim.Adam(policy_model.parameters(), lr=Config.sft_lr)
    policy_model.train()
    
    # 检查是否从文件加载数据
    use_pregenerated_data = sft_data_path is not None and os.path.exists(sft_data_path)
    
    if use_pregenerated_data:
        print(f"    [Data] Loading pre-generated SFT data from: {sft_data_path}")
        data = torch.load(sft_data_path)
        all_problems = data['problems'].to(Config.device)  # [N, tsp_size, 2]
        all_tours = data['tours'].to(Config.device)        # [N, tsp_size]
        num_samples = all_problems.size(0)
        
        print(f"    [Data] Loaded {num_samples} samples")
        print(f"    [Data] TSP Size: {data.get('tsp_size', 'N/A')}")
        print(f"    [Data] Algorithm: {data.get('algorithm', 'N/A')}")
        print(f"    [Data] Avg Length: {data.get('avg_length', 'N/A'):.4f}")
        
        # 计算需要的迭代次数
        num_batches_per_epoch = (num_samples + Config.sft_batch_size - 1) // Config.sft_batch_size
    else:
        print("    [Data] No pre-generated data found, will generate data online.")
        all_problems = None
        all_tours = None
        num_batches_per_epoch = 100  # 默认每个epoch 100个batch
    
    for epoch in range(Config.sft_epochs):
        loss_sum = 0
        pbar = tqdm(range(num_batches_per_epoch), desc=f"SFT Epoch {epoch+1}")
        
        # 如果使用预生成数据，创建随机索引
        if use_pregenerated_data:
            indices = torch.randperm(num_samples, device=Config.device)
        
        for batch_idx in pbar:
            if use_pregenerated_data:
                # 从预生成的数据中采样
                start_idx = batch_idx * Config.sft_batch_size
                end_idx = min(start_idx + Config.sft_batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                x = all_problems[batch_indices]
                target_tours = all_tours[batch_indices]
            else:
                # 在线生成数据（原有逻辑）
                # 1. 生成数据
                x = env.get_random_problems(Config.sft_batch_size, Config.tsp_size)
                
                # 2. 生成标签 (Teacher Output)
                # 在实际科研中，这里通常由 Concorde 或 LKH 离线生成
                with torch.no_grad():
                    target_tours = get_nearest_neighbor_tour(x)
            
            # 3. 计算 Loss (Behavior Cloning)
            # 强制让模型输出 Teacher 的路径
            _, sum_log_probs = policy_model(x, target_tours, teacher_forcing=True)
            
            # Maximize log_prob => Minimize -log_prob
            loss = -sum_log_probs.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            pbar.set_postfix({"SFT Loss": f"{loss.item():.4f}"})
            
    print(">>> SFT Phase Completed. Model now knows basic TSP rules.\n")

    # Save SFT checkpoint for reference
    save_model(policy_model, os.path.join(run_dir, "sft_tsp_model.pth"))
    return policy_model

# ================= Phase 1-4: Iterative DPO Loop =================
def run_iterative_dpo(policy_model, env, run_dir: str):
    print(f">>> Starting Iterative DPO for {Config.total_iterations} iterations...")

    # 固定验证集：降低每轮评估的随机波动（避免误判为“训练震荡”）
    x_eval = None
    if getattr(Config, 'eval_use_fixed_set', False):
        n_eval = int(getattr(Config, 'eval_batch_size', 1000))
        x_eval = env.get_random_problems(n_eval, Config.tsp_size)
        # ========== 加载离线高质量数据（用于混合训练） ==========
    offline_problems = None
    offline_tours = None
    if getattr(Config, 'use_hybrid_data', False):
        hybrid_data_path = getattr(Config, 'hybrid_data_path', None)
        if hybrid_data_path is None:
            hybrid_data_path = Config.sft_data_path
        
        if hybrid_data_path and hybrid_data_path != "None" and os.path.exists(hybrid_data_path):
            print(f"[Hybrid Data] Loading offline high-quality data from: {hybrid_data_path}")
            data = torch.load(hybrid_data_path)
            offline_problems = data['problems'].to(Config.device)
            offline_tours = data['tours'].to(Config.device)
            offline_ratio = getattr(Config, 'hybrid_offline_ratio', 0.3)
            print(f"[Hybrid Data] Loaded {offline_problems.size(0)} samples")
            print(f"[Hybrid Data] Offline ratio: {offline_ratio:.1%} (offline) + {1-offline_ratio:.1%} (online)")
            print(f"[Hybrid Data] Algorithm: {data.get('algorithm', 'N/A')}")
        else:
            print(f"[Hybrid Data] Warning: use_hybrid_data=True but data file not found, using online-only mode")
        # 1. 初始化 Reference Model
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    
    optimizer = optim.Adam(policy_model.parameters(), lr=Config.dpo_lr)
    sampler = PreferenceSampler(policy_model, env)
    
    # 创建温度调度器
    temp_scheduler = create_temperature_scheduler(Config)
    print(f"[Temperature] Strategy: {getattr(Config, 'temperature_decay', 'fixed')}")
    
    history = []
    best_eval_length = float("inf")
    best_model_path = os.path.join(run_dir, "best_tsp_model.pth")

    # 每 10 轮保存一次 checkpoint
    save_every = 10
    
    # 外层循环：迭代轮次 (Generations)
    for iteration in range(Config.total_iterations):
        # 获取当前迭代的温度
        current_temp = temp_scheduler.get_temperature(iteration)
        temp_info = temp_scheduler.get_info(iteration) if hasattr(temp_scheduler, 'get_info') else {}
        
        print(f"\n--- Iteration {iteration+1} / {Config.total_iterations} ---")
        if temp_info:
            print(f"    Temperature: {current_temp:.3f} | Phase: {temp_info.get('phase', 'N/A')} | Progress: {temp_info.get('progress', 0):.1f}%")
        
        # --- Phase 1 & 2: Sampling & Labeling (Self-Generated Data) ---
        # 我们在这里生成这一轮的“经验池”。为了简单，我们每次训练步都实时生成，
        # 但逻辑上属于 Online Data Generation。
        
        policy_model.train()
        iter_loss = 0
        
        # 内层循环：在当前这一批数据分布上进行多次更新
        # 类似于 PPO 的 Epochs per Rollout
        pbar = tqdm(range(Config.epochs_per_iter * 20), desc=f"Iter {iteration+1} Training")
        
        for step in pbar:
            # ========== 混合数据策略：在线生成 + 离线高质量数据 ==========
            use_hybrid = (offline_problems is not None and 
                         getattr(Config, 'use_hybrid_data', False))
            
            if use_hybrid:
                # 计算在线和离线数据的数量
                offline_ratio = getattr(Config, 'hybrid_offline_ratio', 0.3)
                n_offline = int(Config.dpo_batch_size * offline_ratio)
                n_online = Config.dpo_batch_size - n_offline
                
                # A1. 在线生成部分数据
                x_online = env.get_random_problems(n_online, Config.tsp_size)
                x_online, winner_online, loser_online = sampler.sample_dpo_data(
                    x_online, temperature=current_temp
                )
                
                # A2. 从离线数据中随机采样
                indices = torch.randperm(offline_problems.size(0), device=Config.device)[:n_offline]
                x_offline = offline_problems[indices]
                
                # 对离线数据也进行采样（以离线tour为参考生成更多候选）
                # 这样可以利用高质量数据的知识同时保持DPO的对比学习
                with torch.no_grad():
                    # 使用离线tour作为一个候选，再采样其他候选进行对比
                    offline_ref_tours = offline_tours[indices]
                    # 采样额外的候选解
                    x_offline, winner_offline, loser_offline = sampler.sample_dpo_data(
                        x_offline, temperature=current_temp, reference_tour=offline_ref_tours
                    )
                
                # A3. 合并在线和离线数据
                x = torch.cat([x_online, x_offline], dim=0)
                winner_tours = torch.cat([winner_online, winner_offline], dim=0)
                loser_tours = torch.cat([loser_online, loser_offline], dim=0)
                
            else:
                # 原始逻辑：纯在线生成
                # A. 生成问题
                x = env.get_random_problems(Config.dpo_batch_size, Config.tsp_size)
                
                # B. 采样 K 个解并构建 Winner/Loser (Sampler 内部逻辑)
                # 这对应了 Phase 1 (Sampling) 和 Phase 2 (Labeling)
                # 使用动态温度进行采样
                x, winner_tours, loser_tours = sampler.sample_dpo_data(x, temperature=current_temp)

            # 跳过无效偏好对：winner 与 loser 完全相同会导致 logits≈0，训练信号退化为常数
            valid = (winner_tours != loser_tours).any(dim=1)
            if not valid.any():
                continue
            x = x[valid]
            winner_tours = winner_tours[valid]
            loser_tours = loser_tours[valid]
            
            # --- Phase 3: DPO Update ---
            
            # 计算 Policy LogProbs
            _, policy_chosen_logps = policy_model(x, winner_tours, teacher_forcing=True)
            _, policy_rejected_logps = policy_model(x, loser_tours, teacher_forcing=True)
            
            # 计算 Ref LogProbs (No Grad)
            with torch.no_grad():
                _, ref_chosen_logps = ref_model(x, winner_tours, teacher_forcing=True)
                _, ref_rejected_logps = ref_model(x, loser_tours, teacher_forcing=True)

            # 可选：把整条 tour 的 logp sum 归一化为 per-step 平均，降低尺度随 N 增长带来的梯度噪声
            if getattr(Config, 'normalize_logp_by_tour_len', False):
                denom = float(Config.tsp_size)
                policy_chosen_logps = policy_chosen_logps / denom
                policy_rejected_logps = policy_rejected_logps / denom
                ref_chosen_logps = ref_chosen_logps / denom
                ref_rejected_logps = ref_rejected_logps / denom

            loss, loss_val = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=Config.dpo_beta
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 (防止 Transformer 训练不稳定)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            iter_loss += loss_val
            pbar.set_postfix({"DPO Loss": f"{loss_val:.4f}"})
        
        history.append(iter_loss / len(pbar))
        
        # --- Phase 4: Update Reference Model ---
        # 这一步解除了 KL 散度的长期束缚
        if Config.update_ref_model:
            interval = getattr(Config, 'ref_update_interval', 1)
            if interval <= 1 or (iteration + 1) % interval == 0:
                print(">>> Updating Reference Model Weights...")
                ref_model.load_state_dict(policy_model.state_dict())
            
        # Optional: 可以在这里打印一下当前的平均路径长度，看是否有提升
        avg_len = evaluate_model(policy_model, env, x_eval=x_eval)
        if avg_len < best_eval_length:
            best_eval_length = avg_len
            save_model(policy_model, best_model_path)

        if (iteration + 1) % save_every == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_iter{iteration+1}.pth")
            save_model(policy_model, ckpt_path)

    return history

def evaluate_model(model, env, x_eval=None):
    """评估函数：默认随机采样；若传入 x_eval 则复用固定验证集。"""
    model.eval()
    if x_eval is None:
        n_eval = int(getattr(Config, 'eval_batch_size', 100))
        x = env.get_random_problems(n_eval, Config.tsp_size)
    else:
        x = x_eval
    
    # 检查是否使用 POMO 评估
    use_pomo = getattr(Config, 'use_pomo_eval', False)
    
    if use_pomo:
        avg_len = evaluate_pomo(model, env, x)
    else:
        with torch.no_grad():
            # Greedy Decode
            tours, _ = model(x, teacher_forcing=False)
            lengths = env.get_tour_length(x, tours)
        avg_len = lengths.mean().item()
    
    print(f"    [Eval] Avg Tour Length: {avg_len:.4f}")
    return avg_len


def evaluate_pomo(model, env, x, pomo_size=None):
    """POMO 评估：多次采样选最优解
    
    注：标准 POMO 是从不同起点生成路径，但当前模型不支持指定起点。
    这里使用简化版本：多次 greedy 解码（eval 模式下是确定性的），
    或者多次采样（train 模式下是随机的）来近似 POMO 的效果。
    
    Args:
        model: 神经网络模型
        env: TSP 环境
        x: 问题实例 [B, N, 2]
        pomo_size: 采样次数（默认从 Config 读取）
    
    Returns:
        平均路径长度（选择多次采样中的最优解）
    """
    if pomo_size is None:
        pomo_size = getattr(Config, 'pomo_size', 20)
    
    B, N, _ = x.size()
    all_lengths = []
    
    # 临时切换到 train 模式以启用随机采样
    was_training = model.training
    model.train()
    
    # 多次采样
    with torch.no_grad():
        for _ in range(pomo_size):
            tours, _ = model(x, teacher_forcing=False)
            lengths = env.get_tour_length(x, tours)
            all_lengths.append(lengths)
    
    # 恢复原始模式
    model.train(was_training)
    
    # 堆叠所有采样的结果 [B, pomo_size]
    all_lengths = torch.stack(all_lengths, dim=1)
    
    # 每个问题选择最优解
    best_lengths = all_lengths.min(dim=1)[0]
    
    return best_lengths.mean().item()


# ================= Main Entry =================
def main():
    env = TSPEnv(Config.device)

    # 为本次训练创建结果目录
    run_dir = make_run_dir("train", Config.tsp_size)
    print(f"[RunDir] Training outputs will be saved to: {run_dir}")
    
    # 初始化模型（从 Config 读取容量参数）
    embedding_dim = getattr(Config, 'embedding_dim', 128)
    n_encode_layers = getattr(Config, 'n_encode_layers', 3)
    
    policy_model = AttentionModel(
        embedding_dim=embedding_dim, 
        hidden_dim=embedding_dim,  # 保持 hidden_dim = embedding_dim
        n_heads=8, 
        n_encode_layers=n_encode_layers
    ).to(Config.device)
    
    # 1. 运行 SFT (支持从文件加载数据)
    policy_model = run_sft_phase(policy_model, env, run_dir, sft_data_path=Config.sft_data_path)
    
    # 2. 运行 Iterative DPO
    loss_history = run_iterative_dpo(policy_model, env, run_dir)

    # 额外保存最终模型（即使 best 已保存，也方便复现实验）
    save_model(policy_model, os.path.join(run_dir, "final_tsp_model.pth"))
    
    # 3. 绘图
    plt.plot(loss_history, marker='o')
    plt.title("Iterative DPO Loss per Generation")
    plt.xlabel("Iteration")
    plt.ylabel("Avg Loss")
    plt.savefig(os.path.join(run_dir, "iterative_dpo_result.png"))
    print("Done.")

if __name__ == "__main__":
    main()