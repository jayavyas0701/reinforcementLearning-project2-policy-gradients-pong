"""A2C, synchronous advantage actor-critic (Mnih et al., 2016), with n-step returns."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .config import GAMMA, DEVICE, set_seed
from .wrappers import make_env
from .models import ActorCritic, to_tensor
from .logger import RunLogger


def nstep_returns(rewards, dones, last_values, gamma=GAMMA):
    """rewards, dones: arrays of shape (n_steps, n_envs). last_values: V(s) after the last step, shape (n_envs,).
    R_t = r_t + gamma * R_{t+1}, starting from the critic's estimate V(s_n). A finished game stops the chain."""
    returns = np.zeros_like(rewards)
    R = last_values
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * (1.0 - dones[t])
        returns[t] = R
    return returns


def train_a2c(run_name, cfg):
    set_seed(cfg["seed"])
    envs = [make_env() for _ in range(cfg["n_envs"])]
    obs = [env.reset(seed=cfg["seed"] + i)[0] for i, env in enumerate(envs)]
    game_reward = [0.0] * len(envs)
    net = ActorCritic(envs[0].action_space.n).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"])
    log = RunLogger(run_name, net, optimizer, cfg)
    try:
        while not log.should_stop():
            # 1) Play n_steps moves in every env with the current policy.
            batch_obs, batch_actions, batch_rewards, batch_dones = [], [], [], []
            for _ in range(cfg["n_steps"]):
                with torch.no_grad():
                    logits, _ = net(to_tensor(obs))
                actions = Categorical(logits=logits).sample().cpu().numpy()
                rewards, dones, next_obs = [], [], []
                for i, env in enumerate(envs):
                    o, r, terminated, truncated, _ = env.step(actions[i])
                    log.frames += 1
                    game_reward[i] += r
                    done = terminated or truncated
                    if done:
                        log.episode_done(game_reward[i])
                        game_reward[i] = 0.0
                        o, _ = env.reset()
                    rewards.append(r)
                    dones.append(float(done))
                    next_obs.append(o)
                batch_obs.append(obs)
                batch_actions.append(actions)
                batch_rewards.append(rewards)
                batch_dones.append(dones)
                obs = next_obs

            # 2) Targets: n-step returns, bootstrapped from the critic's value of the latest screen.
            with torch.no_grad():
                _, last_values = net(to_tensor(obs))
            returns = nstep_returns(np.array(batch_rewards, dtype=np.float32), np.array(batch_dones, dtype=np.float32),
                                    last_values.cpu().numpy(), cfg["gamma"])

            # 3) One gradient step on the whole batch (n_steps x n_envs transitions).
            x = to_tensor(np.array(batch_obs).reshape(-1, *batch_obs[0][0].shape))
            actions_t = torch.as_tensor(np.array(batch_actions).reshape(-1), device=DEVICE)
            returns_t = torch.as_tensor(returns.reshape(-1), device=DEVICE)
            logits, values = net(x)
            dist = Categorical(logits=logits)
            advantages = returns_t - values.detach()            # how much better than the critic expected
            policy_loss = -(dist.log_prob(actions_t) * advantages).mean()
            value_loss = F.mse_loss(values, returns_t)
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_beta"] * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            optimizer.step()
            log.updates += 1
            if log.updates % 100 == 0:
                log.log("policy_loss", policy_loss.item())
                log.log("value_loss", value_loss.item())
                log.log("entropy", entropy.item())
                log.log("mean_value", values.mean().item())
    finally:                                                # also saves if you stop the cell by hand
        log.close()
        for env in envs:
            env.close()
    return log