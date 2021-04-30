import torch
import numpy as np
from basic_finito import Finito
from torch.autograd import Variable
from linear_regression import LinearRegression

# create dummy data for training
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

inputDim = 1        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 1e4
epochs = 1000

model = LinearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SAG(model.parameters(), lr=learningRate)
optimizer = Finito(model.parameters(), N=x_train.shape[0], lr=learningRate)


for i in range(x_train.shape[0]):
    inputs = Variable(torch.from_numpy(x_train[i, :]))
    labels = Variable(torch.from_numpy(y_train[i, :]))
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.populate_initial_gradients(i)


for epoch in range(epochs):
    # Converting inputs and labels to Variable
    i = optimizer.current_datapoint
    inputs = Variable(torch.from_numpy(x_train[i, :]))
    labels = Variable(torch.from_numpy(y_train[i, :]))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))