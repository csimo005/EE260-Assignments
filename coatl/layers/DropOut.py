import numpy as np
import coatl
from coatl.layers.layer import *

class linear(layer):
    def __init__(self, p=0.5):
        super(layer).__init__()
        self._p = p

    @forward_dec
    def forward(self, x):
        drop = np.random.binomial(2, self._p, x.shape)
        ret_tensor = coatl.tensor(data=(drop*x._data))
        return ret_tensor

    def backward(self, arg0=None, ret0=None):
        grad = ret0._grad/self._p
        arg0.backward(grad)