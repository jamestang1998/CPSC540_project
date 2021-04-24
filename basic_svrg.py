import torch
from base_vr_optimizer import BaseVROptimizer


class SVRG(BaseVROptimizer):

    def __init__(self, params, N, use_numba=False, lr=1e-2, eps=1e-6):
        self.N = N
        self.current_datapoint = -1
        self.passed_samples = 0

        #self.prev_full_grad = []
        #self.previous_parameters = []
        self.t = 10
        self.prev_full_grad = None
        super().__init__(params, use_numba, lr, eps)

    def initialize_state(self, param, state):
        state['step'] = 0
        state['prev'] = [torch.zeros(param.shape)] * self.N  # memory
        state['mean'] = torch.zeros(param.shape)

    def _step(self, model_parameters, optimizer_parameters):
    #def _step(self, model_parameters, optimizer_parameters):
        lr = optimizer_parameters['lr']
         
        self.passed_samples += 1
        for i in range(len(model_parameters['params'])):
            d_p = model_parameters['grads'][i]
        
            prev_d_p = self.previous_parameters[i]

            g_k = d_p - prev_d_p + self.prev_full_grad[i]

            model_parameters['params'][i].add_(g_k, alpha=-lr)
        return

    def store_full_grad(self, layerList):
        self.prev_full_grad = []
        for i in range(len(layerList)):
            self.prev_full_grad.append(layerList[i].grad)
        return

    def store_prev_grad(self, layerList):
        self.previous_parameters = []
        for i in range(len(layerList)):
            self.previous_parameters.append(layerList[i].grad)
        return
    '''
    def store_full_grad(self, model_parameters):
        #print(model_parameters, "model_parameters !!!!!")
        self.prev_full_grad = model_parameters['grads']
        self.zero_grad()
        return
    '''
    def set_step_information(self, info_dict):
        if 'current_datapoint' in info_dict:
            self.current_datapoint = info_dict['current_datapoint']

    def _step_numba(self, params, grads, optimizer_parameters):
        raise NotImplementedError('Not implemented for this class')
