import numpy as np
import coatl
from coatl.layers.layer import *

class relu(layer):
    def __init__(self, alph=0):
        self._alpha = alph
        super(relu).__init__()

    @forward_dec
    def forward(self, x):
        data = np.maximum(x._data, self._alpha*x._data)
        ret_tensor = coatl.tensor(data=data)
        return ret_tensor

    def backward(self, arg0=None, ret0=None):
        grad = np.zeros(arg0.shape)
        grad[arg0._data<=0] = self._alpha
        grad[arg0._data>0] = 1
        grad = grad*ret0._grad
        arg0.backward(grad)