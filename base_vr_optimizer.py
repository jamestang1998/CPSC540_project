import abc
import torch
import numpy as np
from torch.optim import Optimizer


def _transfer_to_shape(data, previous_dims):
    dims = len(data.shape)
    if previous_dims > dims:
        raise ValueError('WTF? previous dim is greater than current dim. {} -> {}'.format(previous_dims, dims))
    if previous_dims == dims:
        return data
    else:
        if previous_dims == 1:
            if dims == 2:
                return data[0, :]
            elif dims == 3:
                return data[0, :, 0]
        elif previous_dims == 2:
            if dims == 3:
                return data[:, :, 0]


class BaseVROptimizer(Optimizer):

    def __init__(self, params, use_numba=True, lr=1e-2, eps=1e-6):
        """
        :param params: The iterative of parameters to be optimized
        :param lr: learning rate of the optimize
        :param eps: epsilon value
        """
        if lr <= 0.:
            raise ValueError("Invalid learning rate: {}, should be >= 0.0".format(lr))
        if eps <= 0.:
            raise ValueError("Invalid epsilon: {}, should be >= 0.0".format(lr))
        defaults = dict(lr=lr, eps=eps)
        self.use_numba = use_numba
        super(BaseVROptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        group = self.param_groups[0]
        model_parameters = {'params': [], 'grads': [], 'states': []}
        optimizer_parameters = group

        for param in group['params']:
            self.initialize_state(param, self.state[param])

        for param in group['params']:
            model_parameters['params'].append(param)
            model_parameters['states'].append(self.state[param])
            model_parameters['grads'].append(None if param.grad is None else param.grad.data)

        if not self.use_numba:
            loss = self._step(model_parameters, optimizer_parameters)
        else:
            params = [p.data.numpy() for p in model_parameters['params']]
            grads = [g.numpy() for g in model_parameters['grads']]
            param_dims = [len(p.shape) for p in params]
            dim_transform_function = {3: np.atleast_3d, 2: np.atleast_2d, 1: np.atleast_1d}[max(param_dims)]
            params = [dim_transform_function(p) for p in params]
            grads = [dim_transform_function(g) for g in grads]
            loss, new_params = self._step_numba(params, grads, optimizer_parameters)
            for i, p in enumerate(new_params):
                model_parameters['params'][i].data = torch.from_numpy(_transfer_to_shape(p, param_dims[i]))
        return loss

    def one_step_GD(self, closure=None):

        group = self.param_groups[0]
        model_parameters = {'params': []}
        optimizer_parameters = group

        for param in group['params']:
            model_parameters['params'].append(param)
        #print(model_parameters)
        if not self.use_numba:
             self._one_step_GD(model_parameters, optimizer_parameters)
        return
    @abc.abstractmethod
    def _step(self, model_parameters, optimizer_parameters):
        pass

    @abc.abstractmethod
    def _step_numba(self, params, grads, optimizer_parameters):
        pass

    @abc.abstractmethod
    def initialize_state(self, param, state):
        pass

    @abc.abstractmethod
    def set_step_information(self, info_dict):
        pass
