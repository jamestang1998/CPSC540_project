import torch
import numpy as np
from batch_sag import BATCH_SAG
from torch.autograd import Variable
from linear_regression import LinearRegression

from torch.utils.data import Dataset
from torchvision import datasets

# create dummy data for training
'''
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
'''
x_train =np.linspace(0,10,999, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [(2.0*i + 5.0) for i in x_train]
y_train= np.array(y_values, dtype=np.float32)#x_train*2+5+np.random.normal(0, 0.1,size=[x_train.shape[0],1])
#y_train = np.array(y_train, dtype=np.float32)

inputDim = 1        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 0.01#0.01
epochs = 5#100

model = LinearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SAG(model.parameters(), lr=learningRate)
optimizer = BATCH_SAG(model.parameters(), N=int(x_train.shape[0]/10), lr=learningRate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

class CustomDataset(Dataset):
    def __init__(self, labels, inputs, transform=None, target_transform=None):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {"inputs": self.labels, "labels": self.inputs}
        return sample
    
train_set = CustomDataset(x_train, y_train)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 10, shuffle=True)

for i in range(x_train.shape[0]):
    inputs = Variable(torch.from_numpy(x_train[i, :]))
    labels = Variable(torch.from_numpy(y_train[i, :]))
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.populate_initial_gradients(i)
    
for epoch in range(epochs):
    
    train_iter = iter(train_loader)
    for itr in range(len(train_loader)):
        # Converting inputs and labels to Variable
        '''
        i = np.random.choice(x_train.shape[0] - 4)
        inputs = Variable(torch.from_numpy(x_train[i, :]))
        labels = Variable(torch.from_numpy(y_train[i, :]))

        inputs1 = Variable(torch.from_numpy(x_train[i + 1, :]))
        labels1 = Variable(torch.from_numpy(y_train[i + 1, :]))

        inputs2 = Variable(torch.from_numpy(x_train[i + 2, :]))
        labels2 = Variable(torch.from_numpy(y_train[i + 2, :]))

        inputs3 = Variable(torch.from_numpy(x_train[i + 3, :]))
        labels3 = Variable(torch.from_numpy(y_train[i + 3, :]))

        inputs4 = Variable(torch.from_numpy(x_train[i + 4, :]))
        labels4 = Variable(torch.from_numpy(y_train[i + 4, :]))



        input_batch = torch.stack((inputs,inputs1, inputs2, inputs3, inputs4), 0)
        label_batch = torch.stack((labels,labels1, labels2, labels3, labels4), 0)
        #print(input_batch.shape, "input batch :)  ")
        #print(label_batch.shape, "label batch :)  ")

        input_batch.requires_grad=True 
        '''
        print(len(x_train), " xtrain shape")
        batch_cpu = next(train_iter)
        print(batch_cpu.keys(), "batch cpu")
        
        inputs = batch_cpu['inputs']
        labels = batch_cpu['labels']
        
        inputs.requires_grad=True 
        #print(inputs, "inputs ")
        #print(labels, "labels ")

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        #outputs = model(inputs)
        outputs = model(inputs)

        # get loss for the predicted output
        #loss = criterion(outputs, labels)
        loss = criterion(inputs, labels)
        #print(loss, "hii")
        print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        #optimizer.set_step_information({'current_datapoint': i})

        # update parameters
        optimizer.step()
        print(list(model.parameters()), " m o d e l !!!")

        print('epoch {}, loss {}'.format(epoch, loss.item()))