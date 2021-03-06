import numpy as np
import coatl
from coatl.layers.layer import *

class linear(layer):
    def __init__(self, input_size, output_size, bias=True, initializer='Gaussian', std=None):
        super(layer).__init__()
        self._bias = bias
        self._in_sz = input_size
        if self._bias:
            self._in_sz += 1
        self._out_sz = output_size

        self._param = coatl.tensor(shape=(self._in_sz, self._out_sz))
        self._param._data = np.random.randn(*self._param.shape)

        if initializer == 'Gaussian':
            self._gaussian_initializer(std)
        elif initializer == 'Zero':
            self._zero_initializer()
        else:
            raise ValueError('Unkown initializer: \'%s\'' % initializer)

    @forward_dec
    def forward(self, x):
        if self._bias: #append 1 feature to x to account for bias
            data = np.concatenate((x._data, np.ones((x.shape[0], 1))), axis=1)@self._param._data
        else:
            data = x._data@self._param._data
        ret_tensor = coatl.tensor(data=data)
        return ret_tensor

    def backward(self, arg0=None, ret0=None):
        if self._bias:
            self._param._grad = np.concatenate((arg0._data, np.ones((arg0.shape[0], 1))), axis=1).T@ret0._grad
            arg0.backward((ret0._grad@self._param._data.T)[:,:-1])
        else:
            self._param._grad = arg0._data.T@ret0._grad
            arg0.backward(ret0._grad@self._param._data.T)

    def get_parameters(self):
        return [self._param]

    def _gaussian_initializer(self, std):
        if std is None:
            std = 1./self._out_sz

        self._param._data = np.random.normal(0., std, self._param.shape)
        return

    def _zero_initializer(self):
        self._param._data = np.zeros(self._param.shape)
        return