"""
================================
Training of the forward MLP model
================================
"""

import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256, 128), dropout=0.0,
                 dom_idx: int = None, alpha: float = 1.0):
        super().__init__()
        # Optionally amplify one selected input dimension.
        self.dom_idx = dom_idx
        self.alpha = alpha

        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.dom_idx is not None and self.alpha != 1.0:
            # Clone the tensor to avoid in-place modification of the input.
            x = x.clone()
            x[..., self.dom_idx] *= self.alpha

        return self.net(x)



