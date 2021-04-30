
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import basic_sag, basic_saga, basic_sgd

torch.manual_seed(0)
batch_size = 1
n_epoch = 2


class CustomRandomSampler:
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, iter_list):
        self.data_source = data_source
        self.iter_list = torch.LongTensor(iter_list)

    def __iter__(self):
        return iter(self.iter_list)

    def __len__(self):
        return len(self.iter_list)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''
        arg : neural net, data
        goal : predict x's classification
        return : classification
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
	'''print image'''
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))


def partial_grad(model, loss_function, data, target, i_data, iter_list):
    """
    arg : neural network, loss function, data,
                target, full_grad (bool), iteration n°i_data, list of initial data randomly picked
    goal : apply partial grad and calculate loss between prediction and result
    return : number of the data, loss
    """
    outputs = model.forward(data)
    loss = loss_function(outputs, target)
    loss.backward()  # store gradient
    return iter_list[i_data], loss

def calculate_loss_grad(model, dataloader, loss_function, iter_list, n_samples):
    """
    inputs : neural network, dataset, loss function, number of samples
    goal : calculate the gradient and the loss
    return : loss and gradient values
    """
    full_loss_epoch = 0
    grad_norm_epoch = 0
    model.zero_grad()
    for i_grad, data_grad in enumerate(dataloader):
        inputs, labels = data_grad
        # inputs, labels = Variable(inputs), Variable(labels)
        # i_data_grad, loss_grad = partial_grad(model, loss_function, inputs, labels, i_grad, iter_list)
        # full_loss_epoch += (1. / n_samples) * loss_grad.data[0]


    return full_loss_epoch


def populate_gradient(model, x, y, index, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.populate_initial_gradients(index)

print('My mom where')

transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5), (0.5))])

trainset = torchvision.datasets.MNIST(root='.', train=True,
										download=True, transform=transform)
n_samples = trainset.data.shape[0]
random_iter = np.random.randint(0, n_samples, n_samples * n_epoch)
sampler = CustomRandomSampler(trainset, random_iter)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler, shuffle=False)

# gradloader = torch.utils.data.DataLoader(trainset, batch_size=1,
# 											shuffle=False, num_workers=2) #to get the gradient for each epoch

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
# 										download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
# 										shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
			'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()


print('I want my mom')


#Define constants
learning_rate = 0.01

#Define nets and parameters
net = Net()
criterion = nn.CrossEntropyLoss()



epoch = 0
running_loss = 0.0
grad_norm_epoch = [0 for i in range(n_epoch)]
full_loss_epoch = [0 for i in range(n_epoch)]


optm = basic_sag.SAG(net.parameters(), N=len(trainloader), lr=learning_rate)

from tqdm import tqdm

print('Trainloader size:', len(trainloader))

# populating initial_grads
for i, data in tqdm(enumerate(trainloader)):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    populate_gradient(net, inputs, labels, i, optm, criterion)

print('Mom?')

for i, data in enumerate(trainloader):

    # full_loss_epoch[epoch] = calculate_loss_grad(net, trainloader, criterion, random_iter, n_samples)
    # epoch += 1
    # get the inputs

    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)

    net.zero_grad()
    i_data, loss = partial_grad(net, criterion, inputs, labels, i, random_iter)
    optm.set_step_information({'current_datapoint': i_data})
    optm.step()

    # print statistics
    running_loss += loss.data[0]
    if i % 2500 == 2499:  # print every 2500 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch, (i) % n_samples + 1, running_loss / 2500))
        running_loss = 0.0
