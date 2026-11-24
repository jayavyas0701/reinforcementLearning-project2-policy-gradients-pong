"""Train REINFORCE on Pong with one of the Project 2 settings (same values as the notebook, Sections 7-8).

Examples:
    python train_reinforce.py --run R2_baseline                     # one 500k-frame tuning case
    python train_reinforce.py --run final                           # the final run: R2 settings, 4M frames
    python train_reinforce.py --run R5_lr_5e-4 --max-frames 20000   # quick smoke test
Re-running the same command resumes from the run's checkpoint.
"""
import argparse, os, shutil
from pg_pong import config
from pg_pong.reinforce import train_reinforce

BASE = dict(seed=config.SEED, n_envs=4, lr=1e-4, gamma=config.GAMMA, baseline=False, reset_on_point=False,
            entropy_beta=0.0, grad_clip=0.5, max_frames=500_000, max_minutes=120)
TUNING = {                                  # R1-R5: each row adds one change to the row above it
    "R1_vanilla":       {},
    "R2_baseline":      dict(baseline=True),
    "R3_point_returns": dict(baseline=True, reset_on_point=True),
    "R4_entropy":       dict(baseline=True, reset_on_point=True, entropy_beta=0.01),
    "R5_lr_5e-4":       dict(baseline=True, reset_on_point=True, entropy_beta=0.01, lr=5e-4),
}
TUNING["R6_batch_16"] = dict(TUNING["R5_lr_5e-4"], n_envs=16)   # R6, R7: start from R5 and change one thing
TUNING["R7_lr_1e-3"] = dict(TUNING["R5_lr_5e-4"], lr=1e-3)
FINAL_FROM = "R2_baseline"                  # best tuning run by final reward_100 (notebook Section 7)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, choices=list(TUNING) + ["final"])
    p.add_argument("--save-dir", default="runs")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--max-minutes", type=float)
    a = p.parse_args()
    config.SAVE_DIR = a.save_dir
    if a.run == "final":
        name, cfg = "reinforce_final", {**BASE, **TUNING[FINAL_FROM], "max_frames": 4_000_000, "max_minutes": 180}
        src = os.path.join(a.save_dir, "reinforce_tuning", FINAL_FROM)
        if not os.path.exists(os.path.join(a.save_dir, name)) and os.path.exists(src):
            shutil.copytree(src, os.path.join(a.save_dir, name))     # continue from the tuning run, as in the notebook
    else:
        name, cfg = f"reinforce_tuning/{a.run}", {**BASE, **TUNING[a.run]}
    if a.max_frames: cfg["max_frames"] = a.max_frames
    if a.max_minutes: cfg["max_minutes"] = a.max_minutes
    print(name, cfg)
    train_reinforce(name, cfg)
