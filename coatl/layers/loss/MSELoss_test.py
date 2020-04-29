import os, sys
sys.path.append(os.path.join(os.getcwd(),'coatl'))

import numpy as np
import tensor
from layers.loss.MSELoss import MSELoss

target = tensor.tensor(data=np.eye(3))
scores = tensor.tensor(data=np.ones((3,3))*0.3)
loss = MSELoss()

l = loss(scores, target)
print(l._data)
print(l._parents)
print(l._kwret)
print(l._operation)

l.backward(tensor.tensor(data=np.asarray([1])))
print(target._grad)
print(scores._grad)

