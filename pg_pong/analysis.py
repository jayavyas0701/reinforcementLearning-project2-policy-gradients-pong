"""Analysis helpers: run summaries, the gradient-noise test (8b) and evaluation with value bias (11)."""
import os
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from . import config
from .config import GAMMA, DEVICE
from .wrappers import make_env
from .models import to_tensor
from .reinforce import discounted_returns


def load_history(run_name):
    """episodes.csv of a run -> DataFrame (episode, frames, minutes, reward, reward_100)."""
    path = os.path.join(config.SAVE_DIR, run_name, "episodes.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if len(df) else None     # None if the run has not finished a game yet


def first_crossing(df, target):
    """First (frames, minutes) at which reward_100 >= target, or (None, None) if never reached."""
    hit = df[df["reward_100"] >= target]
    if hit.empty:
        return None, None
    return int(hit["frames"].iloc[0]), float(hit["minutes"].iloc[0])


def run_summary(name, df):
    row = {"Run": name}
    for target in (0, 15, 19):
        frames, minutes = first_crossing(df, target)
        row[f"Frames to {target}"] = f"{frames/1000:,.0f}k" if frames else "not reached"
        if target == 19:
            row["Minutes to 19"] = f"{minutes:.1f}" if minutes else "not reached"
    row["Best reward_100"] = round(float(df["reward_100"].max()), 2)
    row["Final reward_100"] = round(float(df["reward_100"].iloc[-1]), 2)
    row["Frames trained"] = f"{int(df['frames'].iloc[-1])/1000:,.0f}k"
    row["Minutes trained"] = round(float(df["minutes"].iloc[-1]), 1)
    return row

def batch_gradient(net, games, baseline, reset_on_point):
    """The REINFORCE gradient of one batch of games: the same loss as reinforce_update, but no optimizer step."""
    obs = np.concatenate([np.array(g["obs"]) for g in games])
    actions = torch.as_tensor(np.concatenate([g["actions"] for g in games]), device=DEVICE)
    returns = np.concatenate([discounted_returns(g["rewards"], GAMMA, reset_on_point) for g in games])
    advantages = (returns - returns.mean()) / (returns.std() + 1e-8) if baseline else returns
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
    net.zero_grad()
    for i in range(0, len(obs), 1024):
        x = torch.as_tensor(obs[i:i + 1024], device=DEVICE).float() / 255.0
        log_prob = Categorical(logits=net(x)).log_prob(actions[i:i + 1024])
        (-(log_prob * advantages[i:i + 1024]).sum() / len(obs)).backward()
    return torch.cat([p.grad.flatten() for p in net.parameters()]).cpu()


def agreement(grads):
    """Average cosine similarity between the gradients of different batches (1 = all agree, 0 = random directions)."""
    g = torch.stack(grads)
    g = g / g.norm(dim=1, keepdim=True)
    k = len(grads)
    return float(((g @ g.T).sum() - k) / (k * (k - 1)))     # mean over all pairs, leaving out each batch paired with itself


def evaluate(algo, net, n_games=3, max_steps=20_000):
    """Play n_games and compare the network's own value estimate with the reward it actually got.
    - DQN acts greedily; its value estimate is max_a Q(s, a).
    - A2C samples from its policy (as in training); its value estimate is the critic V(s).
    - REINFORCE samples from its policy; it has no value estimate (it uses the real returns).
    Bias = mean(estimate - real discounted return). Positive bias = over-estimation."""
    env = make_env()
    net.eval()
    scores, estimates, real_returns = [], [], []
    for game in range(n_games):
        obs, _ = env.reset(seed=1000 + game)
        rewards, values = [], []
        for _ in range(max_steps):
            with torch.no_grad():
                out = net(to_tensor([obs]))
            if algo == "dqn":
                action = int(out.argmax(dim=1).item())
                values.append(out.max(dim=1).values.item())
            elif algo == "a2c":
                logits, value = out
                action = int(Categorical(logits=logits).sample().item())
                values.append(value.item())
            else:   # reinforce
                action = int(Categorical(logits=out).sample().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        scores.append(sum(rewards))
        if values:
            estimates.extend(values)
            real_returns.extend(discounted_returns(rewards, GAMMA))
    env.close()
    net.train()
    result = {"algo": algo, "games": n_games, "mean_score": float(np.mean(scores)), "scores": scores}
    if estimates:
        result["mean_estimate"] = float(np.mean(estimates))
        result["mean_real_return"] = float(np.mean(real_returns))
        result["bias"] = result["mean_estimate"] - result["mean_real_return"]
    return result