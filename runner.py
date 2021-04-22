import torch
import matplotlib
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import copy

# For SAG and SAGA
def basic_train(epoch, dataloader, model, use_optimizer, optimizer, criterion, device, model_type, writer=None, update=2000, run_list=None):
    epoch_loss = 0
    running_loss = 0
    count = 0
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()

        index, img, label = data
        index = index.item()

        if model_type == "MLP":
            img = img.view(img.shape[0], -1)
        
        img = img.to(device)
        label = label.to(device)

        output = model(img)

        loss = criterion(output, label)
        
        if use_optimizer in ['SAG', 'SAGA']:
            optimizer.set_step_information({'current_datapoint': index})

        loss.backward()
        optimizer.step()
        
        loss = loss.detach().cpu().item()
        epoch_loss += loss
        running_loss += loss
        count += 1

        if i % update == 0:
            print("Epoch: {} | Iteration {} | Loss: {}".format(epoch, i, running_loss/count))
            writer.add_scalar('Training Loss', running_loss/count, len(run_list)*update)
            run_list.append(running_loss/count)
            running_loss = 0
            count = 0
        
    return {'model': model, 'optimizer': optimizer, 'loss': epoch_loss/len(dataloader), 'run_list': run_list}

# FOR SVRG
def compute_full_grad(model, model_checkpoint, dataloader, model_type, criterion, device, optimizer, optimizer_checkpoint):
    print("Computing Full Gradient")
    # copy the latest "training model"
    model_checkpoint = copy.deepcopy(model)

    # Get the full gradient and store it!
    for i, data in enumerate(dataloader):
        if i % 1000 == 0:
            print('{}/{}'.format(i, len(dataloader)))

        img, label = data

        if model_type == "MLP":
            img = img.view(img.shape[0], -1)
        
        img = img.to(device)
        label = label.to(device)

        output = model_checkpoint(img)

        loss = criterion(output, label)
        loss.backward()
    
    # store into the "main model's" optimizer    
    optimizer.store_full_grad(list(model_checkpoint.parameters()))
    # clear the grads from the checkpoint model
    optimizer_checkpoint.zero_grad()

    return model, model_checkpoint, optimizer, optimizer_checkpoint

# FOR SVRG
def basic_svrg_train(epoch, dataloader, T, current_iteration, model, model_checkpoint, optimizer, optimizer_checkpoint,\
                     criterion, device, model_type, training=True, writer=None, update=2000, run_list=None):

    epoch_loss = 0
    running_loss = 0
    count = 0
    for i, data in enumerate(dataloader):

        if current_iteration % T == 0:
            model, model_checkpoint, optimizer, optimizer_checkpoint = compute_full_grad(model, model_checkpoint, dataloader, model_type,\
                                                                                         criterion, device, optimizer, optimizer_checkpoint)

        optimizer.zero_grad()

        img, label = data

        if model_type == "MLP":
            img = img.view(img.shape[0], -1)

        img = img.to(device)
        label = label.to(device)

        output = model(img)
        checkpoint_output = model_checkpoint(img)

        # get loss for the predicted output
        loss = criterion(output, label)
        checkpoint_loss = criterion(checkpoint_output, label)

        # get gradients w.r.t to parameters
        loss.backward()
        checkpoint_loss.backward()

        # store the current gradients of the checkpoint model
        optimizer.store_prev_grad(list(model_checkpoint.parameters()))

        optimizer.step()

        current_iteration += 1

        loss = loss.detach().cpu().item()
        epoch_loss += loss
        running_loss += loss
        count += 1

        if i % update == 0:
            print("Epoch: {} | Iteration {} | Loss: {}".format(epoch, i, running_loss/count))
            writer.add_scalar('Training Loss', running_loss/count, len(run_list)*update)
            run_list.append(running_loss/count)
            running_loss = 0
            count = 0
        
    return {'model': model, 'optimizer': optimizer, 'model_checkpoint': model_checkpoint,\
            'optimizer_checkpoint':optimizer_checkpoint, 'current_iteration': current_iteration, 'loss': epoch_loss/len(dataloader),\
            'run_list': run_list}