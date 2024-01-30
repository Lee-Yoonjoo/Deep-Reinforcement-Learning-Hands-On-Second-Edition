import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions, dtype=torch.float64)
        )

    def forward(self, x):
        return self.net(x)
