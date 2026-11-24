"""Test games and value-estimate bias (notebook Section 11).

Each best model plays 3 games; for A2C and DQN the network's own value estimate is compared with the discounted
reward that actually followed (gamma = 0.99). Bias = mean(estimate - real return); positive = over-estimation.
    python evaluate.py --save-dir runs --dqn-model PongNoFrameskip-v4-maxboltz-best.dat
The DQN model is the saved Project 1 Max-Boltzmann model (models/ in the Project 1 repository).
"""
import argparse, os
import pandas as pd
import torch
from pg_pong.config import set_seed, SEED, DEVICE
from pg_pong.models import PolicyNet, ActorCritic, DQN
from pg_pong.analysis import evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", default="runs")
    p.add_argument("--dqn-model", default=None)
    a = p.parse_args()
    set_seed(SEED)
    results = []
    net = PolicyNet(6).to(DEVICE)
    net.load_state_dict(torch.load(os.path.join(a.save_dir, "reinforce_final", "best_model.pt"), map_location=DEVICE))
    results.append(evaluate("reinforce", net))
    net = ActorCritic(6).to(DEVICE)
    net.load_state_dict(torch.load(os.path.join(a.save_dir, "a2c_final", "best_model.pt"), map_location=DEVICE))
    results.append(evaluate("a2c", net))
    if a.dqn_model and os.path.exists(a.dqn_model):
        net = DQN((4, 84, 84), 6).to(DEVICE)
        net.load_state_dict(torch.load(a.dqn_model, map_location=DEVICE))
        results.append(evaluate("dqn", net))
    print(pd.DataFrame([{"Algorithm": r["algo"], "Test scores": r["scores"], "Mean test score": round(r["mean_score"], 1),
                         "Bias (estimate - real)": round(r["bias"], 3) if "bias" in r else "n/a (no critic)"}
                        for r in results]).to_markdown(index=False))
