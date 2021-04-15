#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 01:51:48 2021

@author: josef
"""
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
batch_sizes=5
if batch_sizes > x_train.shape[0]:
    print('Warning! You are trying to assign batch greater than number of data.')
    batch_sizes = x_train.shape[0]
epochs = 20

model = LinearRegression(inputDim, outputDim)
model_checkpoint = copy.deepcopy(model)


criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SAG(model.parameters(), lr=learningRate)
optimizer = SARAH(model.parameters(), N=x_train.shape[0],batch_sizes=batch_sizes,lr=learningRate)
#optimizer1 = sarah(model_checkpoint.parameters(), N=x_train.shape[0], lr=learningRate)

for epoch in range(epochs):
    # Converting inputs and labels to Variable

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    model.zero_grad()
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
    model_checkpoint.zero_grad() 
    #print('###### w0 we want ######')
    #print(list(model_checkpoint.parameters()))
    #using GD to update for the first step: w_1
    #print('#######one step GD########')
    optimizer.one_step_GD()
    #print('####### w1 #########')
    #print(list(model.parameters()))
    m = np.random.choice(x_train.shape[0])
    if m==0:
        m=1
    #print('##########inner loop#########')
    for i in range(m):
      ##############################
      #print('######### i={} #########'.format(i))
      # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
      model.zero_grad()
      model_checkpoint.zero_grad()
      
      batch=np.random.choice(a=x_train.shape[0], size=batch_sizes, replace=False)
      for j in batch:
          inputs = Variable(torch.from_numpy(x_train[j, :]))
          labels = Variable(torch.from_numpy(y_train[j, :]))
    
          # get output from the model, given the inputs
          outputs = model(inputs)
          outputs1 = model_checkpoint(inputs)
    
          # get loss for the predicted output
          loss = criterion(outputs, labels)
          loss1 = criterion(outputs1, labels)
    
          # get gradients w.r.t to parameters
          loss.backward()
          loss1.backward()
      
      
      #print('######### w_{} ##########'.format(i))
      #print('w{}'.format(i),list(model.parameters()))
      
      #print('######### gradient of w_{} ##########'.format(i-1))
      optimizer.store_prev_grad(list(model_checkpoint.parameters()))
      model_checkpoint =  copy.deepcopy(model)
      # update parameters
      optimizer.step()
      
      #print('######### w_{} ##########'.format(i+1))
      #print('w{}'.format(i+1),list(model.parameters()))
      
    model_checkpoint =  copy.deepcopy(model)
    
    print('epoch {}, loss {}'.format(epoch, loss.item()))
