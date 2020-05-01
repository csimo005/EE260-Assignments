import os, sys
sys.path.append(os.path.join(os.getcwd()))

import numpy as np

import coatl
from coatl.datasets.mnist_dataset import mnist_dataset
import coatl.layers as layers
import coatl.layers.loss as loss
import coatl.optimizers as optim

import time

class Model(coatl.module):
    def __init__(self):
        self._fc1 = layers.linear(28*28, 100, bias=False)
        self._fc2 = layers.linear(100, 10, bias=False)
        self._activation = layers.sigmoid()
        self._softmax = layers.softmax()

    def forward(self, x):
        x = coatl.tensor(data=np.reshape(x._data, (x.shape[0], 28*28)))
        x = self._activation(self._fc1(x))
        x = self._softmax(self._fc2(x), axis=0)
        return x

def train(model, crit, optimizer, trainloader, epoch, fout=None):
    for i, data in enumerate(trainloader):
        img, label = data

        optimizer.zero_grad()
        scores = model(img)
        loss = crit(scores, label)
        loss.backward()
        optimizer.step()

        if loss._data[0] != loss._data[0]:
            raise ValueError('A wild Nan has appeared, this mean the training has diverged')
        output_str = 'Training[%d, %d] total loss: %.3f' % (epoch, i*trainloader._batch_size, loss._data[0])
        if fout is None:
            print(output_str)
        else:
            fout.write(output_str + '\n')

def test(model, crit, testloader, epoch, fout=None):
    test_loss = 0.
    top_1 = 0.
    total = 0.
    forwards = 0.
    for i, data in enumerate(testloader):
        img, label = data

        scores = model(img)
        loss = crit(scores, label)

        test_loss += loss._data[0]
        total += img.shape[0]
        forwards += 1

        top_1 += coatl.sum(scores.max(axis=1)[1] == label.max(axis=1)[1])
    output_str = 'Testing[%d] test loss: %.3f, accuracy: %.3f' % (epoch, test_loss/forwards, 100*(top_1/total))
    if fout is None:
        print(output_str)
    else:
        fout.write(output_str + '\n')

def main(batchsize=10, lr=0.01, epochs=10, frac_dset=1., fname=''):
    trainset = mnist_dataset(train=True, download=True, frac=frac_dset)
    trainloader = coatl.dataloader(trainset, batchsize)

    testset = mnist_dataset(train=False, download=True)
    testloader = coatl.dataloader(testset, batchsize)

    model = Model()
    optimizer = optim.SGD(model.get_parameters(), lr=lr)
    criterion = loss.LogisticLoss()

    if fname == '':
        fout = None
    else:
        fout = open(fname, 'w')

    t1 = time.time()
    for epoch in range(epochs):
        train(model, criterion, optimizer, trainloader, epoch, fout=fout)
        test(model, criterion, testloader, epoch, fout=fout)
        if epoch%20 == 19:
            optimizer._lr *= 0.1
    t2 = time.time()
    output_str = 'Total Training Time: %.2f seconds' % (t2-t1)
    if fout is None:
        print(output_str)
    else:
        fout.write(output_str+'\n')
        fout.close()

if __name__ == '__main__':
    main(batchsize=100, lr=0.5, epochs=100, fname='mnist_mlp_test.txt')
