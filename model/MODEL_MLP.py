import torch
import torch.nn as nn

import torch.nn.functional as F


class MLP1(nn.Module):
    def __init__(self):
        super(MLP1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(96,400,bias=True),
            nn.ReLU(),
            nn.Linear(400, 400, bias=True),
            nn.ReLU(),
            nn.Linear(400, 400, bias=True),
            nn.ReLU(),
            nn.Linear(400, 400, bias=True),
            nn.ReLU(),
            nn.Linear(400, 400, bias=True),
            nn.ReLU(),
            nn.Linear(400, 400, bias=True),
            nn.ReLU(),
            nn.Linear(400, 400, bias=True),
            nn.ReLU(),
            nn.Linear(400, 400, bias=True),
            nn.ReLU(),        
            nn.Linear(400, 5, bias=True),
        )
    def forward(self, x):
        x = self.model(x)
        return x