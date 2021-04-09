import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(9, 16)
        self.b1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.b2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 4)
        self.b3 = nn.BatchNorm1d(4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):

        x = swish(self.fc1(x))
        # x = self.b1(x)
        x = swish(self.fc2(x))
        # x = self.b2(x)
        x = swish(self.fc3(x))
        # x = self.b3(x)
        x = torch.sigmoid(self.fc4(x))

        return x

