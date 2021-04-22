import torch
import numpy as np
from basic_sag import SAG
from torch.autograd import Variable
from linear_regression import LinearRegression

# create dummy data for training
x_values = [i for i in range(101)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2.0*i + 1.0 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

inputDim = 1        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 0.01
epochs = 100

model = LinearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
optimizer = SAG(model.parameters(), N=x_train.shape[0], lr=learningRate)

for epoch in range(epochs):
    # Converting inputs and labels to Variable
    i = np.random.choice(x_train.shape[0])
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

    #optimizer.set_step_information({'current_datapoint': i})

    # update parameters
    optimizer.step()
    print(list(model.parameters()), " model!")
    print('epoch {}, loss {}'.format(epoch, loss.item()))