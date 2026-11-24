"""Gradient-noise test for REINFORCE (notebook Section 8b).

Plays 128 games with the trained REINFORCE policy, splits them into batches of 4, 8 and 16 games, and reports the
average cosine similarity ("agreement") between the batch gradients: 1 = every batch agrees, 0 = random directions.
    python gradient_noise.py --save-dir runs
"""
import argparse, os
import pandas as pd
import torch
from pg_pong import config
from pg_pong.config import set_seed, SEED, DEVICE
from pg_pong.models import PolicyNet
from pg_pong.wrappers import make_env
from pg_pong.reinforce import collect_episodes
from pg_pong.analysis import batch_gradient, agreement

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", default="runs")
    p.add_argument("--rounds", type=int, default=32, help="rounds of 4 games (32 x 4 = 128 games)")
    p.add_argument("--out", default="results/reinforce_gradient_noise.csv")
    a = p.parse_args()
    set_seed(SEED)
    policy = PolicyNet(6).to(DEVICE)
    policy.load_state_dict(torch.load(os.path.join(a.save_dir, "reinforce_final", "best_model.pt"), map_location=DEVICE))
    envs = [make_env() for _ in range(4)]
    games = []
    for _ in range(a.rounds):
        games += collect_episodes(envs, policy)
    for env in envs:
        env.close()
    print(f"Collected {len(games)} games, {sum(len(g['rewards']) for g in games):,} frames")
    variants = {"R1: vanilla (A = G)": (False, False), "R2: + baseline": (True, False),
                "R3: + baseline + per-point returns": (True, True)}
    rows = []
    for name, (baseline, reset_on_point) in variants.items():
        row = {"Advantage": name}
        for size in (4, 8, 16):
            grads = [batch_gradient(policy, games[i:i + size], baseline, reset_on_point) for i in range(0, len(games), size)]
            row[f"Agreement, {size} games/batch"] = round(agreement(grads), 3)
        rows.append(row)
    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    table.to_csv(a.out, index=False)
    print(table.to_markdown(index=False))
