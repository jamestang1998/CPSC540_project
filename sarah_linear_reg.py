#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 01:51:48 2021

@author: josef
"""
import torch
import numpy as np
from basic_sarah import sarah
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
epochs = 1

model = LinearRegression(inputDim, outputDim)
model_checkpoint = LinearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SAG(model.parameters(), lr=learningRate)
optimizer = sarah(model.parameters(), N=x_train.shape[0], lr=learningRate)
optimizer1 = sarah(model_checkpoint.parameters(), N=x_train.shape[0], lr=learningRate)

for epoch in range(epochs):
    # Converting inputs and labels to Variable

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()
    model_checkpoint.zero_grad()
    # get output from the model, given the inputs
    for i in range(x_train.shape[0]):
      inputs1 = Variable(torch.from_numpy(x_train[i, :]))
      labels1 = Variable(torch.from_numpy(y_train[i, :]))

      outputs1 = model_checkpoint(inputs1)
      loss1 = criterion(outputs1, labels1)
      loss1.backward()

    #optimizer.step()
    optimizer.store_full_grad(list(model_checkpoint.parameters()))
    #update for the first step
    optimizer.one_step_GD()
    m = np.random.choice(x_train.shape[0])
    if m==0:
        m=1
    #store
    for i in range(m):
      ##############################
      j = np.random.choice(m)
      inputs = Variable(torch.from_numpy(x_train[j, :]))
      labels = Variable(torch.from_numpy(y_train[j, :]))

      # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
      model.zero_grad()
      model_checkpoint.zero_grad()
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

      optimizer.store_prev_grad(list(model.parameters()))
      print('v{}k'.format(i),list(model.parameters()))

      optimizer.set_step_information({'current_datapoint': j})
      # update parameters
      optimizer.step()
    model_checkpoint =  copy.deepcopy(model)
    
    print('epoch {}, loss {}'.format(epoch, loss.item()))
