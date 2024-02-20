import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        fx = x.float() / 256
        return self.fc(fx)
