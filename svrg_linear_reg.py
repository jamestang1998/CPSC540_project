import torch
import numpy as np
from basic_svrg import SVRG
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
optimizer = SVRG(model.parameters(), N=x_train.shape[0], lr=learningRate)
optimizer1 = SGD(model_checkpoint.parameters(), lr=learningRate)

for epoch in range(epochs):
    # Converting inputs and labels to Variable

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    #print(x_train.shape, "x train shape")
    for i in range(x_train.shape[0]):
      inputs = Variable(torch.from_numpy(x_train[i, :]))
      labels = Variable(torch.from_numpy(y_train[i, :]))

      outputs = model_checkpoint(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
    
    #paras = list(model_checkpoint.parameters())
    #print(paras)
    #print(model_checkpoint, "HIIII")
    listParam = []
    for i in range(len(list(model_checkpoint.parameters()))):
      #print(list(model_checkpoint.parameters())[i].grad, "HIIII")
      listParam.append(list(model_checkpoint.parameters())[i])
    #print(model_checkpoint.linear.weight.grad, "linear grad")
    #optimizer.store_full_grad(list(model_checkpoint.parameters()))
    optimizer.store_full_grad(listParam)
    #optimizer.store_full_grad(model_checkpoint)
    for i in range(x_train.shape[0]):
      # get loss for the predicted output
      loss = criterion(outputs, labels)
      ##############################
      i = np.random.choice(x_train.shape[0])
      inputs = Variable(torch.from_numpy(x_train[i, :]))
      labels = Variable(torch.from_numpy(y_train[i, :]))

      # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
      #optimizer.zero_grad()

      # get output from the model, given the inputs
      outputs = model(inputs)
      outputs1 = model_checkpoint(inputs)

      # get loss for the predicted output
      loss = criterion(outputs, labels)
      loss1 = criterion(outputs1, labels)

      #print(loss)
      # get gradients w.r.t to parameters
      loss.backward()
      loss1.backward()

      prev_param = []
      for i in range(len(list(model_checkpoint.parameters()))):
        prev_param.append(list(model_checkpoint.parameters())[i])

      optimizer.store_prev_grad(prev_param)

      optimizer.set_step_information({'current_datapoint': i})
      optimizer1.zero_grad()
      # update parameters
      optimizer.step()
    model_checkpoint =  copy.deepcopy(model)

    print('epoch {}, loss {}'.format(epoch, loss.item()))