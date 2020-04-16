import numpy as np

class tensor():
    def __init__(self, shape=(,), dtype=np.float32, data=None):
        self._dtype = dtype
        self.shape = shape
        if data is not None:

        self._data = np.zeros(self.shape, dtype=self._dtype)

        self._require_grad = True
        self._grad = np.zeros(self.shape, dtype=self._dtype)

        self._parents = {}
        self._operation = None
        self._kwret = ''

    def backward(self, grad)
        self._operation.backward(**{**{self._kwret,grad}, **self._parents})

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