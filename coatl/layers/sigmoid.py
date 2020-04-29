import numpy as np
import coatl
from coatl.layers.layer import *

class sigmoid(layer):
    def __init__(self):
        super(sigmoid).__init__()

    @forward_dec
    def forward(self, x):
        data = 1/(1+np.exp(-1*x._data))
        ret_tensor = coatl.tensor(data=data)
        return ret_tensor

    def backward(self, arg0=None, ret0=None):
        sigma = 1 / (1 + np.exp(-1 * arg0._data))
        grad = sigma*(1-sigma)*ret0._grad
        arg0.backward(grad)