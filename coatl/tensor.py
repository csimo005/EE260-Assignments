import numpy as np

class tensor():
    def __init__(self, shape=(), dtype=np.float32, data=None):
        self._dtype = dtype
        self.shape = shape
        if data is not None:
            self._data = np.copy(data)
            self.shape = self._data.shape
            self._dtype = self._data.dtype
        else:
            self.shape = shape
            self._data = np.zeros(self.shape, dtype=self._dtype)

        self._require_grad = True
        self._grad = np.zeros(self.shape, dtype=self._dtype)

        self._parents = {}
        self._operation = None
        self._kwret = ''

    def backward(self, grad=None):
        if grad is not None:
            self._grad = grad
        else:
            self._grad = np.ones(self.shape)
        if len(self._parents):
            self._operation.backward(**{self._kwret:self, **self._parents})

    @property
    def require_grad(self):
        return self._require_grad

    @require_grad.setter
    def require_grad(self, value):
        if type(value) is not bool:
            raise ValueError('Received type: %s, expected bool.' % str(type(value)))
        if value != self._require_grad:
            self._require_grad = value
            if value:
                self._grad = np.zeros(self.shape, dtype=self._dtype)
            else:
                self._grad = None

    def max(self, axis=None):
        ind = np.argmax(self._data, axis=axis)
        if axis is not None:
            val = np.squeeze(np.take_along_axis(self._data, np.expand_dims(ind, axis=axis), axis=axis))
        else:
            val = self._data[np.unravel_index(21, self._data.shape)]
        return tensor(data=val), tensor(data=ind)

    def sum(self, axis=None):
        return sum(self, axis=axis)

    def __eq__(self, other):
        return tensor(data=(self._data == other._data))

def sum(a, axis=None):
    return np.sum(a._data, axis=axis)