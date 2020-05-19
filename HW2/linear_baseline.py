import os, sys
sys.path.append(os.getcwd())
import numpy as np

import coatl
from coatl.datasets.mnist_dataset import mnist_dataset
import coatl.layers as layers
import coatl.layers.loss as loss
import coatl.optimizers as optim
from coatl.utils import dataloader, transforms

import matplotlib.pyplot as plt

import time

class Model(coatl.module):
    def __init__(self):
        self._fc = layers.linear(28*28, 2, initializer='Gaussian', std=1./100)

    def forward(self, x):
        x = x.view((-1,784))
        x = self._fc(x)
        return x

def createLabel(label):
    data = np.zeros((label.shape[0],2))
    data[np.reshape(label._data,(-1,)) <= 4,0] = 1
    data[np.reshape(label._data,(-1,)) > 4,1] = 1
    return coatl.tensor(data=data)

def train(model, crit, optimizer, trainloader, epoch, fout=None):
    temp = np.zeros((28,28))
    for i, data in enumerate(trainloader):
        img, label = data
        label = createLabel(label)

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
        label = createLabel(label)

        scores = model(img)
        loss = crit(scores, label)

        test_loss += loss._data[0]
        total += img.shape[0]
        forwards += 1

        for i in range(img.shape[0]):
            if np.argmax(scores._data[i,:]) == np.argmax(label._data[i,:]):
                top_1 += 1
    output_str = 'Testing[%d] test loss: %.3f, accuracy: %.3f' % (epoch, test_loss/forwards, 100*(top_1/total))
    if fout is None:
        print(output_str)
    else:
        fout.write(output_str + '\n')

def main(batchsize=10, lr=0.01, epochs=10, k=5, frac_dset=1., fname=''):
    tform = transforms.transformation([transforms.ToTensor()])
    trainset = mnist_dataset(train=True, download=True, frac=frac_dset, oneHot=False, transform=tform)
    trainset._images = trainset._images/255.
    mean = np.expand_dims(np.mean(trainset._images, axis=0), 0)
    std = np.expand_dims(np.std(trainset._images, axis=0), 0)
    std[std<=0.03] = 1
    trainset._images = (trainset._images-mean)/std
    trainloader = dataloader(trainset, batchsize)


    testset = mnist_dataset(train=False, download=True, oneHot=False, transform=tform)
    testset._images = testset._images / 255.
    testset._images = (testset._images - mean) / std
    testloader = dataloader(testset, batchsize)

    model = Model()
    optimizer = optim.SGD(model.get_parameters(), lr=lr)
    criterion = loss.MSELoss()

    if fname == '':
        fout = None
    else:
        fout = open(fname, 'w')

    t1 = time.time()
    for epoch in range(epochs):
        train(model, criterion, optimizer, trainloader, epoch, fout=fout)
        test(model, criterion, testloader, epoch, fout=fout)
    t2 = time.time()
    output_str = 'Total Training Time: %.2f seconds' % (t2-t1)
    if fout is None:
        print(output_str)
    else:
        fout.write(output_str+'\n')
        fout.close()

if __name__ == '__main__':
    main(batchsize=1, lr=0.001, epochs=5, frac_dset=5./6, fname='linear_baseline.txt')
