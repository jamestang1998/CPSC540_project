import torch
from base_vr_optimizer import BaseVROptimizer


class BATCH_SAGA(BaseVROptimizer):

    def __init__(self, params, N, batch, use_numba=False, lr=1e-2, eps=1e-6):
        self.N = N
        self.batch = batch
        self.current_datapoint = -1
        self.passed_samples = 0
        super().__init__(params, use_numba, lr, eps)
        for param in self.param_groups[0]['params']:
            self.state[param]['__initial_grad'] = torch.zeros([self.N] + list(param.shape))

    def initialize_state(self, param, state):
        state['step'] = 0
        state['prev'] = state['__initial_grad']  # memory
        state['mean'] = state['__initial_grad'].mean(axis=0)

    def _step(self, model_parameters, optimizer_parameters):
        lr = optimizer_parameters['lr']
        self.passed_samples += self.batch
        for i in range(len(model_parameters['params'])):
            model_parameters['states'][i]['step'] += 1
            d_p = model_parameters['grads'][i].clone()
            if d_p is None:
                continue
            print(self.current_datapoint)
            for j in self.current_datapoint:
                print(j, "hi!!!!!")
                #j = self.current_datapoint  # Todo: extremely hacky, can we improve this?
                mean_grad = model_parameters['states'][i]['mean'].clone()
                prev_grads = model_parameters['states'][i]['prev'][j].clone()
                saga_update = d_p - prev_grads + mean_grad
                model_parameters['states'][i]['mean'] += (1. / int(self.N/self.batch)) * (d_p - prev_grads)
                model_parameters['states'][i]['prev'][j] = d_p
                model_parameters['params'][i].add_(saga_update, alpha=-lr)
        return None

    def set_step_information(self, info_dict):
        if 'current_datapoint' in info_dict:
            self.current_datapoint = info_dict['current_datapoint']

    def _step_numba(self, params, grads, optimizer_parameters):
        raise NotImplementedError('Not implemented for this class')
