from __future__ import annotations

import copy

import torch

from config import Config
from data_sampler import PreferenceSampler
from dpo_loss import dpo_loss
from model import AttentionModel
from tsp_env import TSPEnv


def main() -> None:
    env = TSPEnv(Config.device)

    model = AttentionModel(
        embedding_dim=128,
        hidden_dim=128,
        n_heads=8,
        n_encode_layers=3,
    ).to(Config.device)
    model.train()

    ref = copy.deepcopy(model).eval()
    sampler = PreferenceSampler(model, env)

    x = env.get_random_problems(8, Config.tsp_size)
    x, winner, loser = sampler.sample_dpo_data(x)

    valid = (winner != loser).any(dim=1)
    print("valid_pairs", int(valid.sum().item()), "/", int(valid.numel()))
    if not valid.any():
        print("All pairs degenerate; skipping loss.")
        return

    x = x[valid]
    winner = winner[valid]
    loser = loser[valid]

    _, policy_chosen = model(x, winner, teacher_forcing=True)
    _, policy_rejected = model(x, loser, teacher_forcing=True)

    with torch.no_grad():
        _, ref_chosen = ref(x, winner, teacher_forcing=True)
        _, ref_rejected = ref(x, loser, teacher_forcing=True)

    if getattr(Config, "normalize_logp_by_tour_len", False):
        denom = float(Config.tsp_size)
        policy_chosen = policy_chosen / denom
        policy_rejected = policy_rejected / denom
        ref_chosen = ref_chosen / denom
        ref_rejected = ref_rejected / denom

    loss, loss_val = dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta=Config.dpo_beta,
    )

    loss.backward()
    print("dpo_smoke_ok", loss_val)


if __name__ == "__main__":
    main()
