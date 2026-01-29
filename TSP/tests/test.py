import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from config import Config
from tsp_env import TSPEnv
from model import AttentionModel


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_run_dir(kind: str, tsp_size: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = ensure_dir("result")
    run_dir = os.path.join(base, f"{kind}_{ts}_tsp{tsp_size}")
    ensure_dir(run_dir)
    return run_dir


def find_latest_train_dir(base_dir: str, tsp_size: int) -> str:
    if not os.path.isdir(base_dir):
        return ""
    suffix = f"_tsp{tsp_size}"
    candidates = [
        d
        for d in os.listdir(base_dir)
        if d.startswith("train_") and d.endswith(suffix) and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not candidates:
        # Fall back to latest train run regardless of tsp size
        any_candidates = [
            d
            for d in os.listdir(base_dir)
            if d.startswith("train_") and os.path.isdir(os.path.join(base_dir, d))
        ]
        if not any_candidates:
            return ""
        any_candidates.sort(reverse=True)
        return os.path.join(base_dir, any_candidates[0])
    candidates.sort(reverse=True)  # timestamped names sort lexicographically
    return os.path.join(base_dir, candidates[0])


def load_model(path: str) -> AttentionModel:
    """加载模型权重"""
    print(f"Loading model from {path}...")
    model = AttentionModel(
        embedding_dim=128,
        hidden_dim=128,
        n_heads=8,
        n_encode_layers=3,
    ).to(Config.device)

    if os.path.exists(path):
        state_dict = torch.load(path, map_location=Config.device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: {path} not found. Using random weights.")

    model.eval()
    return model


def plot_tsp(nodes, tour, length: float, title: str, filename: str):
    """可视化 TSP 路径"""
    nodes = nodes.detach().cpu().numpy()
    tour = tour.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=2)

    ordered_nodes = nodes[tour]
    ordered_nodes = np.vstack([ordered_nodes, ordered_nodes[0]])

    plt.plot(ordered_nodes[:, 0], ordered_nodes[:, 1], c="blue", lw=2, alpha=0.7, zorder=1)
    plt.scatter(
        ordered_nodes[0, 0],
        ordered_nodes[0, 1],
        c="green",
        s=100,
        marker="*",
        zorder=3,
        label="Start",
    )

    plt.title(f"{title}\nLength: {length:.4f}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")


def run_sampling_test(model, env, x, num_samples: int = 100):
    """Best-of-K 采样测试：对每个问题采样 K 次，选最短的一条"""
    batch_size, n_nodes, _ = x.size()

    x_repeated = x.repeat_interleave(num_samples, dim=0)

    # 关键点：model.py 里只有 training=True 才会 Categorical 采样
    was_training = model.training
    with torch.no_grad():
        model.train()
        tours_repeated, _ = model(x_repeated, teacher_forcing=False)
    model.train(was_training)

    lengths_repeated = env.get_tour_length(x_repeated, tours_repeated)

    lengths_view = lengths_repeated.view(batch_size, num_samples)
    tours_view = tours_repeated.view(batch_size, num_samples, n_nodes)

    best_lengths, best_indices_in_group = lengths_view.min(dim=1)
    best_tours = tours_view.gather(
        1,
        best_indices_in_group.view(batch_size, 1, 1).expand(-1, -1, n_nodes),
    ).squeeze(1)

    return best_lengths, best_tours


def main():
    parser = argparse.ArgumentParser(description="Test a trained TSP model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to a .pth weight file (overrides auto-latest).",
    )
    args = parser.parse_args()

    env = TSPEnv(Config.device)

    # 为本次测试创建结果目录
    test_dir = make_run_dir("test", Config.tsp_size)
    print(f"[RunDir] Test outputs will be saved to: {test_dir}")

    # 1. 加载模型（默认加载最新一次训练的 best）
    if args.model_path:
        model_path = args.model_path
    else:
        latest_train_dir = find_latest_train_dir("result", Config.tsp_size)
        if latest_train_dir:
            model_path = os.path.join(latest_train_dir, "best_tsp_model.pth")
        else:
            model_path = "best_tsp_model.pth"
    model = load_model(model_path)

    # 2. 生成测试集（数量不宜太多，因为采样会消耗显存）
    num_test = 50
    print(f"\nGenerating {num_test} test problems (TSP-{Config.tsp_size})...")
    x = env.get_random_problems(num_test, Config.tsp_size)

    # ================= 模式 1: Greedy（基准） =================
    print("\n>>> Running Greedy Search (Baseline)...")
    model.eval()
    with torch.no_grad():
        greedy_tours, _ = model(x, teacher_forcing=False)
        greedy_lengths = env.get_tour_length(x, greedy_tours)

    greedy_avg = greedy_lengths.mean().item()
    print(f"Greedy Avg Length: {greedy_avg:.4f}")

    # ================= 模式 2: Sampling（进阶） =================
    sample_k = 100
    print(f"\n>>> Running Sampling Strategy (Best of {sample_k})...")
    sampling_lengths, sampling_tours = run_sampling_test(model, env, x, num_samples=sample_k)

    sampling_avg = sampling_lengths.mean().item()
    print(f"Sampling Avg Length: {sampling_avg:.4f}")

    # ================= 结果对比与绘图 =================
    improvement = (greedy_avg - sampling_avg) / greedy_avg * 100
    print(f"\n>>> Improvement: {improvement:.2f}%")

    # 保存指标
    with open(os.path.join(test_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"model_path={model_path}\n")
        f.write(f"tsp_size={Config.tsp_size}\n")
        f.write(f"num_test={num_test}\n")
        f.write(f"sample_k={sample_k}\n")
        f.write(f"greedy_avg={greedy_avg:.6f}\n")
        f.write(f"sampling_avg={sampling_avg:.6f}\n")
        f.write(f"improvement_pct={improvement:.4f}\n")

    diff = greedy_lengths - sampling_lengths
    best_idx = torch.argmax(diff).item()
    print(f"Visualizing result for problem index {best_idx}...")

    plot_tsp(
        x[best_idx],
        greedy_tours[best_idx],
        greedy_lengths[best_idx].item(),
        title="Greedy Strategy",
        filename=os.path.join(test_dir, "test_compare_greedy.png"),
    )

    plot_tsp(
        x[best_idx],
        sampling_tours[best_idx],
        sampling_lengths[best_idx].item(),
        title=f"Sampling Strategy (Best of {sample_k})",
        filename=os.path.join(test_dir, "test_compare_sampling.png"),
    )


if __name__ == "__main__":
    main()
