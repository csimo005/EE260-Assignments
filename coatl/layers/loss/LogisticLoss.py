import numpy as np
import coatl
from coatl.layers.layer import *

class LogisticLoss(layer):
    def __init__(self, average=True, eps=1e-6):
        super(LogisticLoss).__init__()
        self._average = average
        self._eps = eps

    @forward_dec
    def forward(self, activations, target):
        if np.sum(activations._data<0):
            raise ValueError('Logistic Loss function can not take negative values.')
        l = -np.sum(target._data*np.log(activations._data+self._eps) + (1-target._data)*np.log(1-activations._data+self._eps))
        l = l/activations.shape[1]
        if self._average:
            l = l / activations.shape[0]
        return coatl.tensor(data=np.asarray([l]))

    def backward(self, arg0, arg1, ret0):
        grad = ((1-arg1._data)/(1-arg0._data+self._eps)) - (arg1._data/(arg0._data+self._eps))

        grad = grad/arg0.shape[1]
        if self._average:
            grad = grad/arg0.shape[0]
        arg0.backward(grad)