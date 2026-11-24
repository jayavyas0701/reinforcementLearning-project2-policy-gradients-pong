"""RunLogger: reward_100, TensorBoard, episodes.csv, best model, checkpoint every 10 minutes and resume."""
import os
import time
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import config as settings


class RunLogger:
    """Book-keeping shared by REINFORCE and A2C:
    - keeps every finished game's reward and the mean of the last 100 (reward_100)
    - writes TensorBoard scalars and an episodes.csv file
    - saves the best model, and a checkpoint every few minutes so a Colab disconnect can be resumed
    If a checkpoint already exists for this run, training continues from it.
    A run stops at reward_100 >= settings.TARGET_REWARD, at max_frames (total), or after max_minutes in this session."""

    def __init__(self, run_name, net, optimizer, config):
        self.net, self.optimizer, self.config = net, optimizer, config
        self.dir = os.path.join(settings.SAVE_DIR, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.dir, "checkpoint.pt")
        self.best_path = os.path.join(self.dir, "best_model.pt")
        self.frames, self.updates, self.best_reward_100 = 0, 0, -float("inf")
        self.history = []            # one row per finished game: (episode, frames, minutes, reward, reward_100)
        self.minutes_before = 0.0    # training minutes from earlier sessions (when resuming)
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=settings.DEVICE, weights_only=False)
            net.load_state_dict(ckpt["net"])
            optimizer.load_state_dict(ckpt["optimizer"])
            self.frames, self.updates = ckpt["frames"], ckpt["updates"]
            self.best_reward_100, self.history = ckpt["best_reward_100"], ckpt["history"]
            self.minutes_before = ckpt["minutes"]
            print(f"Resumed {run_name}: {self.frames:,} frames, {len(self.history)} games, "
                  f"{self.minutes_before:.1f} min done before")
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        self.writer = SummaryWriter(os.path.join(self.dir, "tb"))
        self.start = time.time()
        self.last_save = self.last_print = time.time()

    def minutes(self):
        return self.minutes_before + (time.time() - self.start) / 60

    def reward_100(self):
        return self.history[-1][4] if self.history else -21.0

    def episode_done(self, reward):
        last_rewards = [row[3] for row in self.history[-99:]] + [reward]
        r100 = float(np.mean(last_rewards))
        self.history.append((len(self.history) + 1, self.frames, round(self.minutes(), 2), reward, round(r100, 3)))
        self.writer.add_scalar("reward", reward, self.frames)
        self.writer.add_scalar("reward_100", r100, self.frames)
        if r100 > self.best_reward_100:
            self.best_reward_100 = r100
            torch.save(self.net.state_dict(), self.best_path)
        if time.time() - self.last_print > 60:     # one status line per minute
            fps = self.frames / max(1e-9, self.minutes() * 60)
            print(f"{self.frames:>10,} frames | {len(self.history):>5} games | reward_100 {r100:6.2f} | "
                  f"best {self.best_reward_100:6.2f} | {self.minutes():6.1f} min | {fps:4.0f} frames/s")
            self.last_print = time.time()

    def log(self, name, value):
        self.writer.add_scalar(name, value, self.frames)

    def should_stop(self):
        if len(self.history) >= 1 and self.reward_100() >= settings.TARGET_REWARD:
            print(f"Converged: reward_100 = {self.reward_100():.2f} at {self.frames:,} frames, {self.minutes():.1f} min")
            return True
        if self.frames >= self.config["max_frames"]:
            print(f"Reached the frame budget ({self.config['max_frames']:,} frames)")
            return True
        if (time.time() - self.start) / 60 >= self.config["max_minutes"]:    # time limit for this session only
            print(f"Reached this session's time limit ({self.config['max_minutes']} min). Run the cell again to continue.")
            return True
        if time.time() - self.last_save > 600:     # checkpoint every 10 minutes
            self.save()
        return False

    def save(self):
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "frames": self.frames, "updates": self.updates, "best_reward_100": self.best_reward_100,
                    "history": self.history, "minutes": self.minutes()}, self.ckpt_path)
        with open(os.path.join(self.dir, "episodes.csv"), "w") as f:
            f.write("episode,frames,minutes,reward,reward_100\n")
            for row in self.history:
                f.write(",".join(str(v) for v in row) + "\n")
        self.last_save = time.time()

    def close(self):
        self.save()
        self.writer.close()
        print(f"Saved to {self.dir}: {self.frames:,} frames, {len(self.history)} games, "
              f"best reward_100 {self.best_reward_100:.2f}, {self.minutes():.1f} min")