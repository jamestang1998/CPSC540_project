import torch
import numpy as np
import basic_sag, basic_saga, batch_sag, batch_saga
from torch.autograd import Variable
from linear_regression import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import random
import torch
random.seed(10)
torch.manual_seed(10)

# create dummy data for training
x_values = [i for i in range(10)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2.0*i + 1.0 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

inputDim = 1        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 1e-3
epochs = 800

model = LinearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
optimizer = batch_saga.BATCH_SAGA(model.parameters(), N=x_train.shape[0], batch = 2, lr=learningRate)

for i in range(x_train.shape[0]):
    inputs = Variable(torch.from_numpy(x_train[i, :]))
    labels = Variable(torch.from_numpy(y_train[i, :]))
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.populate_initial_gradients(i)

indices = list(range(x_train.shape[0]))

for epoch in range(epochs):
    if epoch % x_train.shape[0] == 0:
        random.shuffle(indices)
    # Converting inputs and labels to Variable
    i = indices[epoch % x_train.shape[0]]
    inputs = Variable(torch.from_numpy(x_train[i, :]))
    labels = Variable(torch.from_numpy(y_train[i, :]))
    
    inputs1 = Variable(torch.from_numpy(x_train[(i + 1) % x_train.shape[0], :]))
    labels1 = Variable(torch.from_numpy(y_train[(i + 1) % x_train.shape[0], :]))
    
    input_batch = torch.stack((inputs,inputs1), 0)
    label_batch = torch.stack((labels,labels1), 0)

    input_batch.requires_grad=True 

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    #outputs = model(inputs)
    output_batch = model(input_batch)

    # get loss for the predicted output
    #loss = criterion(outputs, labels)
    loss = criterion(output_batch, label_batch)
    # get gradients w.r.t to parameters
    loss.backward()

    optimizer.set_step_information({'current_datapoint': [i, (i + 1) % x_train.shape[0]]})

    # update parameters
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

print(list(map(lambda x: x.detach().numpy(), model.parameters())))