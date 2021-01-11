import torch
from torch import nn


class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, batchnorm, dropout):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(output_size) if batchnorm else nn.Identity(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.model(x)


class Baseline(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        linear_size,
        num_layers,
        batchnorm=False,
        dropout=0,
    ):
        super().__init__()
        self.model = nn.Sequential(
            *[
                FullyConnected(input_size, linear_size, batchnorm, dropout)
                if (i == 0)
                else FullyConnected(linear_size, linear_size, batchnorm, dropout)
                for i in range(num_layers)
            ],
            nn.Linear(linear_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.model(input)
