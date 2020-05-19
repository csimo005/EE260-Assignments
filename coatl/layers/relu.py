import numpy as np
import coatl
from coatl.layers.layer import *

class relu(layer):
    def __init__(self, alpha=0):
        self._alpha = alpha
        super(relu).__init__()

    @forward_dec
    def forward(self, x):
        data = x._data*(np.ones(x.shape) - (x._data<=0)*(1-self._alpha))
        ret_tensor = coatl.tensor(data=data)
        return ret_tensor

    def backward(self, arg0=None, ret0=None):
        grad = np.ones(arg0.shape) - (arg0._data<=0)*(1-self._alpha)
        grad = grad*ret0._grad
        arg0.backward(grad)