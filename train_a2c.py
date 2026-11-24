"""Train A2C on Pong with one of the Project 2 settings (same values as the notebook, Sections 9-10).

Examples:
    python train_a2c.py --run A1_textbook               # 2M-frame tuning case
    python train_a2c.py --run A2_lr_2e-3 --max-frames 4000000   # equal-budget re-test (Section 10b)
    python train_a2c.py --run final                     # textbook settings until reward_100 >= 19 (14M-frame cap)
Re-running the same command resumes from the run's checkpoint.
"""
import argparse, os, shutil
from pg_pong import config
from pg_pong.a2c import train_a2c

BASE = dict(seed=config.SEED, n_envs=32, n_steps=4, lr=1e-3, adam_eps=1e-3, gamma=config.GAMMA, value_coef=1.0,
            entropy_beta=0.01, grad_clip=0.1, max_frames=2_000_000, max_minutes=150)
TUNING = {"A1_textbook": {}, "A2_lr_2e-3": dict(lr=2e-3), "A3_clip_0.5": dict(grad_clip=0.5)}
FINAL_FROM = "A1_textbook"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, choices=list(TUNING) + ["final"])
    p.add_argument("--save-dir", default="runs")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--max-minutes", type=float)
    a = p.parse_args()
    config.SAVE_DIR = a.save_dir
    if a.run == "final":
        name, cfg = "a2c_final", {**BASE, **TUNING[FINAL_FROM], "max_frames": 14_000_000, "max_minutes": 300}
        src = os.path.join(a.save_dir, "a2c_tuning", FINAL_FROM)
        if not os.path.exists(os.path.join(a.save_dir, name)) and os.path.exists(src):
            shutil.copytree(src, os.path.join(a.save_dir, name))
    else:
        name, cfg = f"a2c_tuning/{a.run}", {**BASE, **TUNING[a.run]}
    if a.max_frames: cfg["max_frames"] = a.max_frames
    if a.max_minutes: cfg["max_minutes"] = a.max_minutes
    print(name, cfg)
    train_a2c(name, cfg)
