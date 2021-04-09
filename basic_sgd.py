from base_vr_optimizer import BaseVROptimizer
import numba as nb


@nb.njit
def optimize_sgd(params, grads, lr):
    for i in range(len(params)):
        params[i] += -lr * grads[i]
    return 0, params


class NumbaSGD(BaseVROptimizer):

    def _step_numba(self, params, grads, optimizer_parameters):
        return optimize_sgd(params, grads, optimizer_parameters['lr'])


class SGD(BaseVROptimizer):

    def initialize_state(self, param, state):
        state['step'] = 0

    def set_step_information(self, info_dict):
        pass

    def _step(self, model_parameters, optimizer_parameters):
        lr = optimizer_parameters['lr']
        # model_parameters = {'params': [1, ], 'grads': [-.1, ], 'states': [{}, ]}
        for i in range(len(model_parameters['params'])):
            d_p = model_parameters['grads'][i]
            if d_p is not None:
                model_parameters['params'][i].add_(d_p, alpha=-lr)  # calculating the parameter
                model_parameters['states'][i]['step'] += 1  # just a state
        return None

    def _step_numba(self, params, grads, optimizer_parameters):
        return optimize_sgd(params, grads, optimizer_parameters['lr'])
