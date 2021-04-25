import os
import torch.optim as optim
from basic_svrg import SVRG
from basic_sag import SAG
from basic_saga import SAGA
from basic_sarah import SARAH
from class_saga import Class_SAGA
from model import MLP, CNN, RNN

def build_model(model_name, dataset_name, device):
    if dataset_name == 'MNIST':
        input_size = 28*28*1
        output_size = 10
        input_channels = 1
        intermediate_size = 128
        if model_name == 'MLP':
            model = MLP(input_size, output_size)
        else: # model_name == 'CNN':
            model = CNN(input_channels, intermediate_size, output_size)

    else: # dataset_name = CIFAR
        input_size = 32*32*3
        output_size = 10
        input_channels = 3
        intermediate_size = 200
        if model_name == 'MLP':
            model = MLP(input_size, output_size)
        else:# model_name == 'CNN':
            model = CNN(input_channels, intermediate_size, output_size)
    return model.to(device)

def build_optimizer(optimizer_name, model, lr, N):
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SAG':
        optimizer = SAG(model.parameters(), N=N, lr=lr)
    elif optimizer_name == 'SAGA':
        optimizer = SAGA(model.parameters(), N=N, lr=lr)
    elif optimizer_name == "SVRG":
        optimizer = SVRG(model.parameters(), N=N, lr=lr)
    elif optimizer_name == "SARAH":
        optimizer = SARAH(model.parameters(), N=N, batch_sizes = 1, lr=lr)
    elif optimizer_name == "Class_SAGA":
        optimizer = Class_SAGA(model.parameters(), N=N, lr=lr)
    else:
        optimizer = None
        assert optimizer != None
    return optimizer

def create_folder(root, run):
    folder = os.path.join(root, run)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def save_args(path, args):
    with open(path, 'w') as f:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            f.write('{} = {}\n'.format(arg, attr))

def save_configs(path, config):
    with open(path, 'w') as f:
        f.write(open(config, 'r').read())
        
def generate_unique_run_name(name, model_save_path, run_save_path):
    run_string = "_run="
    run_count = 0
    not_unique = True
    new_run_name = name + run_string
    while not_unique:
        temp_new_run_name = new_run_name + str(run_count)
        temp_model_save_path = os.path.join(model_save_path, temp_new_run_name)
        temp_run_save_path = os.path.join(run_save_path, temp_new_run_name)
        if os.path.exists(temp_model_save_path) or os.path.exists(temp_run_save_path):
            run_count += 1
        else:
            new_run_name = temp_new_run_name
            not_unique = False
    return new_run_name
    