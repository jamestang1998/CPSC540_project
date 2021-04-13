import torch
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch import optim
from basic_svrg import SVRG
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from model import MLP, CNN, RNN

import matplotlib.pyplot as plt
import numpy as np
import copy

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

n_epochs = 3
batch_size_train = 16
learning_rate = 0.01
T = 3


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = MLP(32*32*3, 10)
model_checkpoint = MLP(32*32*3, 10)


criterion = nn.CrossEntropyLoss()
optimizer = SVRG(model.parameters(), N=len(trainset), lr=learning_rate)
optimizer_checkpoint = SVRG(model_checkpoint.parameters(), N=len(trainset), lr=learning_rate)

for epoch in range(n_epochs):
    if epoch % T == 0:
        print("Computing Full Gradient")
        # copy the latest "training model"
        model_checkpoint = copy.deepcopy(model)

        # Get the full gradient and store it!
        for i, data in enumerate(trainloader):
            if i % 1000 == 0:
                print('{}/{}'.format(i, len(trainloader)))
            inputs, labels = data
            flatten_image = inputs.view(-1, 32*32*3)

            predicted_class = model_checkpoint(flatten_image)

            loss = criterion(predicted_class, labels)
            loss.backward()

        # store into the "main model's" optimizer    
        optimizer.store_full_grad(list(model_checkpoint.parameters()))
        # clear the grads from the checkpoint model
        optimizer_checkpoint.zero_grad()

    print("Inner Training Loop")
    for i, data in enumerate(trainloader):
        
        optimizer.zero_grad()

        inputs, labels = data
        flatten_image = inputs.view(-1, 32*32*3)

        output = model(flatten_image)
        checkpoint_output = model_checkpoint(flatten_image)

        # get loss for the predicted output
        loss = criterion(output, labels)
        checkpoint_loss = criterion(checkpoint_output, labels)

        # get gradients w.r.t to parameters
        loss.backward()
        checkpoint_loss.backward()

        # store the current gradients of the checkpoint model
        optimizer.store_prev_grad(list(model_checkpoint.parameters()))

        optimizer.step()
        
        if i % 1000 == 0:
            print("Epoch {} | Iteration {} | loss {}".format(epoch, i, loss.detach().item()))





