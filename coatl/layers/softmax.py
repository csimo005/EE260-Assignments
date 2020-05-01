import numpy as np
import coatl
from coatl.layers.layer import *

class softmax(layer):
    def __init__(self):
        super(softmax).__init__()

    @forward_dec
    def forward(self, x, axis=None):
        if axis is not None and not isinstance(axis, int):
            raise TypeError('Error expected param axis to be type int, got %s' % str(type(axis)))
        if axis is not None and axis>=len(x.shape):
            raise ValueError('Specified axis \'%d\', which is greater than max dim of x \'%d\'' % (axis, len(x.shape)-1))
        ax = np.arange(len(x.shape))
        if axis is not None:
            ax = np.delete(ax, axis)
        exp = np.exp(x._data)

        z = np.sum(exp, axis=tuple(ax))
        tile_sz = list(x._data.shape)
        z_sz = [1]*len(tile_sz)
        tile_sz[axis], z_sz[axis] = z_sz[axis], tile_sz[axis]
        z = np.tile(np.reshape(z, tuple(z_sz)), tuple(tile_sz))

        ret_tensor = coatl.tensor(data=exp/z)
        return ret_tensor

    def backward(self, arg0=None, axis=None, ret0=None):
        ax = np.arange(len(ret0.shape))
        if axis is not None:
            ax = np.delete(ax, axis)

        sm = np.sum(ret0._data*ret0._grad, tuple(ax))
        tile_sz = list(ret0._data.shape)
        z_sz = [1] * len(tile_sz)
        tile_sz[axis], z_sz[axis] = z_sz[axis], tile_sz[axis]
        grad = ret0._data*(ret0._grad - np.tile(np.reshape(sm, tuple(z_sz)), tuple(tile_sz)))

        arg0.backward(grad)