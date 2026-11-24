"""Settings shared by every run (same as Project 1). SAVE_DIR can be changed by the scripts (--save-dir)."""
import random
import numpy as np
import torch

ENV_NAME = "PongNoFrameskip-v4"   # same environment as Project 1
GAMMA = 0.99                      # same discount as Project 1
TARGET_REWARD = 19.0              # "converged" = mean of last 100 games >= 19 (same as Project 1)
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "runs"                 # every run is saved here; the notebook uses a Google Drive folder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
