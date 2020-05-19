import numpy as np

class SGD():
    def __init__(self, parameters, lr=0.1):
        self._parameters = parameters
        self._lr = lr

    def step(self):
        for param in self._parameters:
            param._data = param._data - self._lr*param._grad

    def zero_grad(self):
        for param in self._parameters:
            param._grad[:] = np.zeros(param.shape)