import os, sys
sys.path.append(os.path.join(os.getcwd()))

import coatl
import coatl.layers as layers

X = coatl.tensor(shape=(1,10))
W = layers.linear(10,5,bias=False)
Y = W(X)
print(Y.shape)