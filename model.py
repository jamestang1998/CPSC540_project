import torch 
import torch.nn as nn
import torch.nn.functional as F

# Code copied from: https://github.com/kilianFatras/variance_reduced_neural_networks/blob/master/SAGA.ipynb
class CNN(nn.Module):
    def __init__(self, num_channels, intermediate_size, output_dim):
        super(CNN, self).__init__()
        self.intermediate_size = intermediate_size
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
<<<<<<< HEAD
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
=======
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(intermediate_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)
        self.softmax = nn.Softmax(dim=1)
>>>>>>> 8fe07e01974bd65d8dc1db30b85e90a7ef5402ba

    def forward(self, x):
        '''
        arg : neural net, data
        goal : predict x's classification
        return : classification
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
# Code copied from: https://github.com/kilianFatras/variance_reduced_neural_networks/blob/master/SAGA.ipynb
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 84)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        arg : neural net, data
        goal : predict x's classification
        return : classification
        '''
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc3(x))
        # x = self.fc3(x)
        x = self.softmax(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(RNN, self).__init__()
        self.rnn = nn.rnn(input_dim, hidden_dim, num_layers)
        self.fc1 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = F.relu(self.fc1(x))
        return x