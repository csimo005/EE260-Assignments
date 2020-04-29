import numpy as np
import coatl
from coatl.layers.layer import *

class MSELoss(layer):
    def __init__(self, average=True):
        super(MSELoss).__init__()
        self._average = average

    @forward_dec
    def forward(self, activations, target):
        l = 0.
        for i in range(activations.shape[0]):
            l += 0.5*np.sum((target._data[i,:]-activations._data[i,:])**2)
        if self._average:
            l = l/(activations.shape[0])
        return coatl.tensor(data=np.asarray([l]))

    def backward(self, arg0, arg1, ret0):
        grad = arg0._data-arg1._data
        if self._average:
            grad = grad/arg0.shape[0]
        grad = grad*ret0._data[0]
        arg0.backward(grad)