import torch
import random
import numpy as np
from base_vr_optimizer import BaseVROptimizer


def get_index_of_unknown_shaped_tensor(data, index):
    if len(data.shape) == 1:
        return data[index]
    elif len(data.shape) == 2:
        return data[index, :]
    elif len(data.shape) == 3:
        return data[index, :, :]
    elif len(data.shape) == 4:
        return data[index, :, :, :]
    else:
        raise ValueError("Bro what is that parameter, stop, it has a shape of {}!".format(str(data.shape)))


# Todo: care, this needs sampling without replacement
class Finito(BaseVROptimizer):

    def __init__(self, params, N, use_numba=False, lr=1e-2, eps=1e-6):
        self.N = N
        self.current_datapoint = -1
        self.passed_samples = set()
        self.random_points = list(range(N))
        random.shuffle(self.random_points)
        self.n_index = 0
        self.current_datapoint = self.random_points[self.n_index]
        self.passed_samples.add(self.current_datapoint)
        super().__init__(params, use_numba, lr, eps)
        for param in self.param_groups[0]['params']:
            self.state[param]['__initial_grad'] = torch.zeros([self.N] + list(param.shape))

    def initialize_state(self, param, state):
        # Todo: this param is a value that might be initialized randomly.
        #  Should we use that value for the selected datapoint instead of zeros?
        state['step'] = 0
        state['grads'] = state['__initial_grad']  # memory for gradients
        state['values'] = torch.zeros([self.N] + list(param.shape))  # memory for parameter values

    def _step(self, model_parameters, optimizer_parameters):
        lr = optimizer_parameters['lr']
        for i in range(len(model_parameters['params'])):
            d_p = model_parameters['grads'][i]
            if d_p is None:
                continue
            model_parameters['states'][i]['grads'][self.current_datapoint] = d_p
        if self.n_index == self.N - 1:
            random.shuffle(self.random_points)
        self.n_index = (self.n_index + 1) % self.N
        next_datapoint = self.random_points[self.n_index]
        for i in range(len(model_parameters['params'])):
            model_parameters['states'][i]['step'] += 1
            grads = model_parameters['states'][i]['grads']
            param_values = model_parameters['states'][i]['values']
            if len(self.passed_samples) < self.N:
                param_avg = param_values[list(self.passed_samples)].mean(axis=0)
                grads_avg = grads[list(self.passed_samples)].mean(axis=0)
            else:
                param_avg = param_values.mean(axis=0)
                grads_avg = grads.mean(axis=0)
            model_parameters['params'][i] = param_avg - (1. / lr) * grads_avg
            param_values[next_datapoint] = model_parameters['params'][i]
        self.passed_samples.add(next_datapoint)
        self.current_datapoint = next_datapoint
        return None

    def _step_numba(self, params, grads, optimizer_parameters):
        raise NotImplementedError('Not implemented for this class')
