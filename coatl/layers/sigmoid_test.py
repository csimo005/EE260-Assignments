import os, sys
sys.path.append(os.path.join(os.getcwd(),'coatl'))

import numpy as np
import tensor
from layers.loss.MSELoss import MSELoss
from layers.linear import linear
from layers.sigmoid import sigmoid

xData = np.zeros((9,5))
for i in range(5):
    xData[:,i] = np.linspace(-1,1,9)**(i+1)
print(xData)
X = tensor.tensor(data=xData)

yData = (xData @ np.asarray(([[1], [0.3], [2], [0], [-0.5]]))) + 5
yData = 1./(1+np.exp(-1*yData))
Y = tensor.tensor(data=yData)

criterion = MSELoss(average=False)
fc = linear(5, 1)
sig = sigmoid()

for i in range(100000):
    yHat = sig(fc(X))
    loss = criterion(yHat, Y)
    print(loss._data)

    loss.backward(np.asarray([1]))
    fc._param._data -= 0.1*fc._param._grad

print(yHat._data)
print(yData)
print(fc._param._data)