"""Networks: the same conv layers as the Project 1 DQN for every algorithm."""
import numpy as np
import torch
import torch.nn as nn

from .config import DEVICE


class ConvBody(nn.Module):
    """The same 3-layer conv stack as the Project 1 DQN (Mnih et al., 2015). Input (batch, 4, 84, 84)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.out_size = 64 * 7 * 7   # = 3136 features for an 84x84 input

    def forward(self, x):
        return self.conv(x)


class PolicyNet(nn.Module):
    """REINFORCE network: screen -> one score (logit) per action. Softmax of the logits = the policy."""
    def __init__(self, n_actions):
        super().__init__()
        self.body = ConvBody()
        self.policy = nn.Sequential(nn.Linear(self.body.out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        return self.policy(self.body(x))


class ActorCritic(nn.Module):
    """A2C network: shared conv body, then an actor head (action logits) and a critic head (state value V(s)).
    Same layout as the textbook's AtariA2C (Lapan, Ch. 10)."""
    def __init__(self, n_actions):
        super().__init__()
        self.body = ConvBody()
        self.policy = nn.Sequential(nn.Linear(self.body.out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))
        self.value = nn.Sequential(nn.Linear(self.body.out_size, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        features = self.body(x)
        return self.policy(features), self.value(features).squeeze(-1)


class DQN(nn.Module):
    """Project 1 DQN, copied unchanged, only so we can load the saved Project 1 model for the comparison."""
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


def to_tensor(obs_list):
    """List of observations -> float tensor on the GPU."""
    return torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=DEVICE)