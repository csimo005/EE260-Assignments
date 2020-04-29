import numpy as np
from .layers import layer

class module():
    def __init__(self):
        return

    def forward(self, x):
        raise NotImplementedError

    def get_parameters(self):
        param = []
        for k in self.__dict__.keys():
            if isinstance(self.__dict__[k],layer.layer):
                param += self.__dict__[k].get_parameters()
        return param

    def train(self):
        for item in dir(self).items():
            if type(item) is layer:
                layer.train()

    def eval(self):
        for item in dir(self).items():
            if type(item) is layer:
                layer.eval()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)