import numpy as np
import torch
from torch import nn


class LunarA2c(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(LunarA2c, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        fc_out_size = self._get_fc_out(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(fc_out_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(fc_out_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def _get_fc_out(self, shape):
        o = self.fc(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float()
        fc_out = self.fc(fx).view(fx.size()[0], -1)
        return self.policy(fc_out), self.value(fc_out)
