import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 4)
        # self.fc2 = nn.Linear(16, 8)
        # self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.sig(self.fc1(x))
        # x = self.activate(self.fc2(x))
        # x = self.activate(self.fc3(x))
        x = self.sig(self.fc4(x))
        return x
