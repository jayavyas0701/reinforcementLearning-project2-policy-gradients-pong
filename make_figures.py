"""Rebuild the report figures from runs/*/episodes.csv.   Usage: python make_figures.py runs results/figures"""
import sys, os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RUNS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150, "savefig.dpi": 200,
                     "axes.grid": True, "grid.alpha": 0.3, "font.family": "DejaVu Sans"})
load = lambda r: pd.read_csv(os.path.join(RUNS, r, "episodes.csv"))
TARGET = 19
C = {"REINFORCE": "#1f77b4", "A2C": "#ff7f0e", "DQN": "#2ca02c", "DQNb": "#7f7f7f"}

# Fig 1: REINFORCE tuning, all 7 cases
names = {"R1_vanilla": "R1 vanilla", "R2_baseline": "R2 + baseline", "R3_point_returns": "R3 + per-point returns",
         "R4_entropy": "R4 + entropy bonus", "R5_lr_5e-4": "R5 lr 5e-4", "R6_batch_16": "R6 R5 + 16 games/update",
         "R7_lr_1e-3": "R7 R5 + lr 1e-3"}
fig, ax = plt.subplots(figsize=(8, 3.6))
for k, lab in names.items():
    d = load(f"reinforce_tuning/{k}")
    ax.plot(d.frames / 1e3, d.reward_100, lw=1.2, label=lab)
ax.set_ylim(-21.2, -19.6); ax.set_xlabel("Frames (thousands)"); ax.set_ylabel("Mean reward (last 100 games)")
ax.set_title("REINFORCE tuning runs (500k frames each)"); ax.legend(ncol=2, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_reinforce_tuning.png"); plt.close(fig)

# Fig 2: REINFORCE final run, 4M frames
d = load("reinforce_final")
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.scatter(d.frames / 1e6, d.reward, s=3, alpha=0.18, color=C["REINFORCE"], label="Score of each game")
ax.plot(d.frames / 1e6, d.reward_100, color="#0b3d6b", lw=1.6, label="Mean of last 100 games")
ax.axvline(0.503, color="gray", ls=":", lw=1); ax.text(0.53, -15.4, "end of R2 tuning run", fontsize=8, color="gray")
ax.set_ylim(-21.5, -14.5); ax.set_xlabel("Frames (millions)"); ax.set_ylabel("Reward")
ax.set_title("Final REINFORCE run (R2 settings, 4M frames)"); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_reinforce_final.png"); plt.close(fig)

# Fig 3: A2C tuning at an equal 4M budget
fig, ax = plt.subplots(figsize=(8, 3.6))
for lab, run in [("A1 textbook (lr 1e-3, clip 0.1)", "a2c_final"), ("A2 lr 2e-3", "a2c_tuning/A2_lr_2e-3"),
                 ("A3 clip 0.5", "a2c_tuning/A3_clip_0.5")]:
    d = load(run); d = d[d.frames <= 4_000_000]
    ax.plot(d.frames / 1e6, d.reward_100, lw=1.3, label=lab)
ax.axvline(2.0, color="gray", ls=":", lw=1); ax.text(2.03, -17.75, "first tuning budget (2M)", fontsize=8, color="gray")
ax.axhline(-19, color="gray", ls="--", lw=0.8)
ax.set_xlabel("Frames (millions)"); ax.set_ylabel("Mean reward (last 100 games)")
ax.set_title("A2C tuning runs at an equal 4M-frame budget"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_a2c_tuning_4M.png"); plt.close(fig)

# Fig 4: final A2C run with milestones
d = load("a2c_final")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.scatter(d.frames / 1e6, d.reward, s=3, alpha=0.12, color=C["A2C"], label="Score of each game")
ax.plot(d.frames / 1e6, d.reward_100, color="#a14d00", lw=1.6, label="Mean of last 100 games")
for t in (-19, 0, 15, 19):
    h = d[d.reward_100 >= t].iloc[0]
    ax.plot(h.frames / 1e6, t, "ko", ms=4)
    ax.annotate(f"{t} at {h.frames/1e6:.2f}M", (h.frames / 1e6, t), xytext=(-78, 6), textcoords="offset points", fontsize=8)
ax.axhline(TARGET, color="gray", ls="--", lw=0.8)
ax.set_xlabel("Frames (millions)"); ax.set_ylabel("Reward"); ax.set_ylim(-22, 22)
ax.set_title("Final A2C run (textbook settings) until the mean reaches 19"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_a2c_final.png"); plt.close(fig)

# Fig 5: comparison vs Project 1 DQN (milestones at reward_100 = 0, 15, 19 from Project 1 milestones.csv)
DQN = {"DQN ε-greedy (Project 1 baseline)": ([293, 390, 862], [45.1, 60.9, 138.8]),
       "DQN Max-Boltzmann (Project 1 best)": ([331, 424, 492], [50.6, 65.3, 75.8])}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for algo, run in [("REINFORCE", "reinforce_final"), ("A2C", "a2c_final")]:
    d = load(run)
    axes[0].plot(d.frames / 1e6, d.reward_100, color=C[algo], lw=1.4, label=algo)
    axes[1].plot(d.minutes, d.reward_100, color=C[algo], lw=1.4, label=algo)
for (lab, (fr, mn)), col, mk in zip(DQN.items(), [C["DQNb"], C["DQN"]], ["s--", "o--"]):
    axes[0].plot(np.array(fr) / 1e3, [0, 15, 19], mk, color=col, ms=4, lw=1, label=lab)
    axes[1].plot(mn, [0, 15, 19], mk, color=col, ms=4, lw=1, label=lab)
for ax, xl, t in zip(axes, ["Frames (millions)", "Training time (minutes)"], ["Reward vs frames", "Reward vs training time"]):
    ax.axhline(TARGET, color="gray", ls="--", lw=0.8); ax.set_xlabel(xl); ax.set_ylabel("Mean reward (last 100 games)"); ax.set_title(t)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUT}/fig5_comparison.png"); plt.close(fig)
print("figures written to", OUT)
