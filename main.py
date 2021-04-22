import os
import torch
import matplotlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
import configargparse
import utils
import runner

from model import MLP, CNN, RNN


def config_parser():
    parser = configargparse.ArgumentParser()
    
    # Load config
    parser.add_argument('--config', is_config_file=True,
                        help='path to config file')

    # Hyperparameters and Dataset
    parser.add_argument('--epoch', type=int, default=5,
                        help='total number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate during training')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training/evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for the dataloaders')     
    parser.add_argument('--seed', type=int, default=1,
                        help='seed to use')
    parser.add_argument('--use_dataset', type=str,
                        help='which dataset to use', default='CIFAR')
    parser.add_argument('--use_optimizer', type=str,
                        help='which optimizer to use', default='SGD')
    parser.add_argument('--use_model', type=str,
                        help='which model to use', default='MLP')

    # SVRG
    parser.add_argument('--T', type=int, default=1,
                        help='T to use')
    # GPU Allocation
    parser.add_argument('--cuda_num', type=int, default=0,
                        help='which GPU to use if there are multiple available')

    # Logging/Monitoring
    parser.add_argument('--run_folder', type=str,
                        help='folder to store run files', default="logs")
    parser.add_argument('--model_path', type=str,
                        help='where to store model checkpoints', default='model')
    
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    total_epochs = args.epoch
    learning_rate = args.lr
    batch_size = args.batch_size
    use_dataset = args.use_dataset
    num_workers = args.num_workers
    run_folder = args.run_folder
    model_path = args.model_path
    use_optimizer = args.use_optimizer
    use_model = args.use_model
    T = args.T

    assert use_dataset in ['MNIST', 'CIFAR']
    assert use_optimizer in ['SGD', 'SAG', 'SAGA', 'SVRG', 'SARAH', 'FINITO']  
    assert use_model in ['MLP', 'CNN', 'RNN']
    
    assert T > 0

    torch.manual_seed(args.seed)

    # device = torch.device("cuda:{}".format(args.cuda_num) if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    run_name = "{}-{}-{}".format(use_optimizer, use_model, use_dataset)
    run_name = utils.generate_unique_run_name(run_name, model_path, run_folder)
    
    # Create folders for logging
    model_folder = utils.create_folder(model_path, run_name)
    run_folder = utils.create_folder(run_folder, run_name)
    
    utils.save_args(os.path.join(run_folder, "args.txt"), args)
    utils.save_configs(os.path.join(run_folder, "config.txt"), args.config)
    writer = SummaryWriter(os.path.join(run_folder, "total_epoch={}-bs={}-lr={}".format(total_epochs, batch_size, learning_rate)))

    print('INITIALIZING DATASET')
    if use_dataset == 'CIFAR':
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./local/tangytob/CPSC540/data', train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root='./local/tangytob/CPSC540/data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)
    
    else: # it's MNIST
        transform = transforms.Compose(
                    [transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='./local/tangytob/CPSC540/data', train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=0)

        testset = torchvision.datasets.MNIST(root='./local/tangytob/CPSC540/data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)
    
    print('Training Examples: {}'.format(len(trainset)))
    print('Test Examples: {}'.format(len(testset)))
            
    # Initialize Models
    print('INITIALIZING MODELS')
    model = utils.build_model(use_model, use_dataset, device)
    if use_optimizer == 'SVRG':
        model_checkpoint = utils.build_model(use_model, use_dataset, device)
    else:
        model_checkpoint = None
    
    print('INITIALIZING OPTIMIZER')
    optimizer = utils.build_optimizer(use_optimizer, model, learning_rate, len(trainset))
    if use_optimizer == 'SVRG':
        optimizer_checkpoint = utils.build_optimizer(use_optimizer, model_checkpoint, learning_rate, len(trainset))
    else:
        optimizer_checkpoint = None
    
    lowest_validation_loss = 1e7

    criterion = nn.CrossEntropyLoss()

    current_iteration = 0

    print('Starting Training')
    for epoch in range(total_epochs):
        
        # Training loop
        if use_optimizer != "SVRG":
            train_dict = runner.basic_train(epoch, trainloader, model, optimizer, criterion, device, use_model, writer, update=100)
            model = train_dict['model']
            optimizer = train_dict['optimizer']
            training_loss = train_dict['loss']
        else:
            train_dict = runner.basic_svrg_train(epoch, trainloader, T, current_iteration, model, model_checkpoint, optimizer, optimizer_checkpoint,\
                                                 criterion, device, use_model, writer, update=100)
            model = train_dict['model']
            optimizer = train_dict['optimizer']
            model_checkpoint = train_dict['model_checkpoint']
            optimizer_checkpoint = train_dict['optimizer_checkpoint']
            training_loss = train_dict['loss']

        """SAVE MODEL AND OPTIMIZER"""
        training_file = os.path.join(model_folder, "latest_epoch.tar")
        torch.save({
                    'epoch': epoch,
                    'batch_size': batch_size,
                    # 'validation_loss': validation_loss,
                    'lowest_validation_loss':lowest_validation_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
        }, training_file)

    print('Finished Training')