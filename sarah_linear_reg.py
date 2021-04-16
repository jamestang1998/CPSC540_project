import torch
import numpy as np
from basic_sarah import SARAH
from torch.autograd import Variable
from linear_regression import LinearRegression

from basic_sgd import NumbaSGD, SGD

import copy

# create dummy data for training
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

inputDim = 1        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 0.01
epochs = 100

model = LinearRegression(inputDim, outputDim)
model_checkpoint = LinearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SAG(model.parameters(), lr=learningRate)
optimizer = SARAH(model.parameters(), N=x_train.shape[0], lr=learningRate)
optimizer1 = SARAH(model_checkpoint.parameters(), N=x_train.shape[0], lr=learningRate)

from random import randrange

for epoch in range(epochs):
    # Converting inputs and labels to Variable

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()
    optimizer1.zero_grad()

    # get output from the model, given the inputs
    for i in range(x_train.shape[0]):
      inputs = Variable(torch.from_numpy(x_train[i, :]))
      labels = Variable(torch.from_numpy(y_train[i, :]))

      outputs = model_checkpoint(inputs)
      loss = criterion(outputs, labels)
      loss.backward()

    optimizer.store_full_grad(list(model_checkpoint.parameters()))

    t_rand = randrange(x_train.shape[0]) #random model to be chosen after inner loop
    wt_model = None
    for i in range(x_train.shape[0]):
      ##############################
      j = np.random.choice(x_train.shape[0])
      inputs = Variable(torch.from_numpy(x_train[j, :]))
      labels = Variable(torch.from_numpy(y_train[j, :]))

      # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
      optimizer.zero_grad()

      # get output from the model, given the inputs
      outputs = model(inputs)
      outputs1 = model_checkpoint(inputs)

      # get loss for the predicted output
      loss = criterion(outputs, labels)
      loss1 = criterion(outputs1, labels)

      if i == t_rand:
        wt_model = copy.deepcopy(model)

      #print(loss)
      # get gradients w.r.t to parameters
      loss.backward()
      loss1.backward()

      optimizer.store_prev_grad(list(model_checkpoint.parameters()))

      optimizer.set_step_information({'current_datapoint': i})
      # update parameters
      optimizer.step()
    model_checkpoint =  copy.deepcopy(wt_model)
    
    print('epoch {}, loss {}'.format(epoch, loss.item()))