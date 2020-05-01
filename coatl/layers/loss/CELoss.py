import numpy as np
import coatl
from coatl.layers.layer import *

class CELoss(layer):
    def __init__(self, average=True):
        super(CELoss).__init__()
        self._average = average

    @forward_dec
    def forward(self, activations, target):
        l = -np.sum(np.log(activations._data[np.arange(activations.shape[0]), target._data]))
        if self._average:
            l = l/(activations.shape[0])
        return coatl.tensor(data=np.asarray([l]))

    def backward(self, arg0, arg1, ret0):
        grad = np.zeros(arg0._data.shape)
        grad[np.arange(arg0.shape[0]), arg1._data] = -1/arg0._data[np.arange(arg0.shape[0]), arg1._data]
        if self._average:
            grad = grad/arg0.shape[0]
        arg0.backward(grad)