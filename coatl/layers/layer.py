import os, sys
import numpy as np
import coatl
import time

def forward_dec(func):
    def wrapper(*args, **kwargs):
        args_d = {}
        for i in range(1,len(args)):
            args_d['arg%d'%(i-1)] = args[i]
        parents = {**args_d, **kwargs}
        ret_val = func(*args, **kwargs)
        if type(ret_val) is not tuple:
            ret_val._parents = parents
            ret_val._kwret = 'ret0'
            ret_val._operation = args[0]
        else:
            for i in range(len(ret_val)):
                if type(ret_val) is coatl.tensor:
                    ret_val[i]._parents = parents
                    ret_val[i]._kwret = 'ret%d' % i
                    ret_val._operation = args[0]
        return ret_val
    return wrapper

class layer():
    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def get_parameters(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)