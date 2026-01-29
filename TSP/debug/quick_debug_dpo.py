from __future__ import annotations

import os
from datetime import datetime

import torch

from config import Config
from tsp_env import TSPEnv
from model import AttentionModel
from train import run_iterative_dpo


def make_run_dir(kind: str, tsp_size: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "result"
    os.makedirs(base, exist_ok=True)
    run_dir = os.path.join(base, f"{kind}_{ts}_tsp{tsp_size}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main() -> None:
    # ===== 临时覆盖：只用于快速诊断，不影响你正式训练 =====
    Config.total_iterations = 10
    Config.epochs_per_iter = 1  # run_iterative_dpo 内部会 *20 -> 20 steps
    Config.dpo_batch_size = 32
    Config.num_samples = 6

    # 更频繁打印诊断信息
    Config.dpo_log_stats = True
    Config.dpo_log_every_steps = 1

    # 更稳的 ref 更新频率
    Config.update_ref_model = True
    Config.ref_update_interval = 5

    # 保持 per-step 归一化
    Config.normalize_logp_by_tour_len = True

    print(
        f"[DebugConfig] tsp_size={Config.tsp_size} batch={Config.dpo_batch_size} K={Config.num_samples} "
        f"iters={Config.total_iterations} steps/iter={Config.epochs_per_iter*20}"
    )

    env = TSPEnv(Config.device)
    run_dir = make_run_dir("debug_dpo", Config.tsp_size)

    policy_model = AttentionModel(
        embedding_dim=128,
        hidden_dim=128,
        n_heads=8,
        n_encode_layers=3,
    ).to(Config.device)

    # 让采样更随机一些（你的 model 里 training=True 才会 Categorical 采样）
    policy_model.train()

    # 跑短版 DPO，观察 tqdm postfix 中的 valid/gap/logits
    _ = run_iterative_dpo(policy_model, env, run_dir)

    # 额外检查梯度是否正常（防止 NaN）
    total_norm = 0.0
    with torch.no_grad():
        for p in policy_model.parameters():
            if p.grad is not None:
                total_norm += float(p.grad.detach().data.norm(2).cpu())
    print(f"[GradCheck] total_grad_l2_sum={total_norm:.4f}")


if __name__ == "__main__":
    main()
