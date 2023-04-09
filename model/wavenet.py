from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import *

    
class ResidualBlock(nn.Module):
    def __init__(self, residual_channels):        
        super().__init__()
        self.dilated_conv = nn.Conv1d(residual_channels, residual_channels, 1, dilation=2)
        self.middle_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x):
        y = self.dilated_conv(x)
        y = self.middle_projection(y)

        gate, fltr = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(fltr)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class WaveNet(BASE):
    def __init__(self, **model_args):
        super().__init__(model_args['lr'])
        seq_len = model_args['seq_len']
        n_features = model_args["n_features"]
        residual_channels = model_args["residual_channels"]
        residual_layers = model_args["residual_layers"]
        
        self.input_projection = nn.Sequential(
            nn.Conv1d(seq_len, residual_channels, 1),
            nn.LeakyReLU()
        )
        
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels) for _ in range(residual_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(residual_channels * n_features, residual_channels//4 * n_features),
            nn.BatchNorm1d(residual_channels//4 * n_features),
            nn.LeakyReLU(),
            nn.Linear(residual_channels//4 * n_features, 1)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        skip = 0
        for layer in self.residual_layers:
            x, skip_connection = layer(x)
            skip = skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.output_projection(x)
        return x.reshape(x.shape[0], 1, 1)