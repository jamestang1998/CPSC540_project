import torch
from base_vr_optimizer import BaseVROptimizer
import copy



class SAGA(BaseVROptimizer):

    def __init__(self, params, N, use_numba=False, lr=1e-2, eps=1e-6):
        self.N = N
        self.current_datapoint = -1
        self.passed_samples = 0
        super().__init__(params, use_numba, lr, eps)

    def initialize_state(self, param, state):
        state['step'] = 0
        state['prev'] = [torch.zeros(param.shape)] * self.N  # memory
        state['mean'] = torch.zeros(param.shape)

    def _step(self, model_parameters, optimizer_parameters):
        lr = optimizer_parameters['lr']
        self.passed_samples += 1
        for i in range(len(model_parameters['params'])):
            model_parameters['states'][i]['step'] += 1
            d_p = model_parameters['grads'][i]
            if d_p is None:
                continue
            mean_grad = model_parameters['states'][i]['mean']
            prev_grads = model_parameters['states'][i]['prev']
            j = self.current_datapoint  # Todo: extremely hacky, can we improve this?
            saga_update = d_p - prev_grads[j] + mean_grad
            mean_grad += (1. / self.N) * (copy.deepcopy(d_p) - prev_grads[j])
            prev_grads[j] = copy.deepcopy(d_p)
            model_parameters['params'][i].add_(saga_update, alpha=-lr)
        return None

    def set_step_information(self, info_dict):
        if 'current_datapoint' in info_dict:
            self.current_datapoint = info_dict['current_datapoint']

    def _step_numba(self, params, grads, optimizer_parameters):
        raise NotImplementedError('Not implemented for this class')
