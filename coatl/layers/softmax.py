import numpy as np
import coatl
from coatl.layers.layer import *

class softmax(layer):
    def __init__(self):
        super(sigmoid).__init__()

    @forward_dec
    def forward(self, x):
        exp = np.exp(x._data)
        ret_tensor = coatl.tensor(data=exp/np.sum(exp))
        return ret_tensor

    def backward(self, arg0=None, ret0=None):

        arg0.backward(grad)