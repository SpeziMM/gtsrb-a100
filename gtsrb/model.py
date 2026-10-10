import torch.nn as nn


class Net(nn.Module):
    """The Assignment 5 architecture, unchanged: 151,403 parameters."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.linear = nn.Linear(1024, 43)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.pool1(self.elu(self.conv1(x)))
        x = self.pool2(self.elu(self.conv2(x)))
        return self.linear(x.flatten(1))
