import torch
from base_vr_optimizer import BaseVROptimizer


class SAG(BaseVROptimizer):

    def __init__(self, params, N, use_numba=False, lr=1e-2, eps=1e-6):
        self.N = N
        self.current_datapoint = -1
        self.passed_samples = 0
        super().__init__(params, use_numba, lr, eps)
        for param in self.param_groups[0]['params']:
            self.state[param]['__initial_grad'] = torch.zeros([self.N] + list(param.shape))

    def initialize_state(self, param, state):
        state['step'] = 0
        state['Y'] = state['__initial_grad']  # memory n * d
        state['D'] = state['__initial_grad'].mean(axis=0)

    def _step(self, model_parameters, optimizer_parameters):
        lr = optimizer_parameters['lr']
        self.passed_samples += 1
        for i in range(len(model_parameters['params'])):
            model_parameters['states'][i]['step'] += 1
            d_p = model_parameters['grads'][i].clone()
            if d_p is None:
                continue
            D = model_parameters['states'][i]['D'].clone()
            j = self.current_datapoint  # Todo: extremely hacky, can we improve this?
            Y = model_parameters['states'][i]['Y'][j].clone()
            D = D - (1./self.N) * (Y - d_p)
            model_parameters['states'][i]['D'] = D
            model_parameters['states'][i]['Y'][j] = d_p
            model_parameters['params'][i].add_(D, alpha=-lr/min(self.N, self.passed_samples))
#             model_parameters['params'][i].add_(D, alpha=-lr/self.N)
        return None

    def _one_step_GD(self, model_parameters, optimizer_parameters):
        pass

    def set_step_information(self, info_dict):
        if 'current_datapoint' in info_dict:
            self.current_datapoint = info_dict['current_datapoint']

    def _step_numba(self, params, grads, optimizer_parameters):
        raise NotImplementedError('Not implemented for this class')
