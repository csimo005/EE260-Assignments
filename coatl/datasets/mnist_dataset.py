import os
import requests
import gzip
import struct
import numpy as np

import coatl

def read_idx(fname):
    dtype_lut = [()]*16
    dtype_lut[8] = (np.uint8, 1, lambda s: struct.unpack('>%dB' % len(s), s))
    dtype_lut[9] = (np.int8, 1, lambda s: struct.unpack('>%db' % len(s), s))
    dtype_lut[11] = (np.int16, 2, lambda s: struct.unpack('>%dh' % len(s)//2, s))
    dtype_lut[12] = (np.int32, 4, lambda s: struct.unpack('>%di'% len(s)//4, s))
    dtype_lut[13] = (np.float32, 4, lambda s: struct.unpack('>%df' % len(s)//4, s))
    dtype_lut[14] = (np.float64, 8, lambda s: struct.unpack('>%dd'% len(s)//8, s))

    with gzip.open(fname, 'rb') as fin:
        magic_num = fin.read(4)
        if dtype_lut[magic_num[2]] == ():
            raise RuntimeError('Undefined datatype in idx magic number: %d' % magic_num[2])

        size = struct.unpack('>%dL' % magic_num[3], fin.read(4*magic_num[3]))
        numel = 1
        for i in range(1,len(size)):
            numel*=size[i]
        data = np.empty(tuple(size), dtype_lut[magic_num[2]][0])
        print('0/%d' % size[0], end='')
        for i in range(size[0]):
            values = dtype_lut[magic_num[2]][2](fin.read(dtype_lut[magic_num[2]][1]*numel))
            if len(size) > 1:
                values = np.asarray(values, dtype=dtype_lut[magic_num[2]][0])
                data[i] = np.reshape(values, tuple(size[1:]))
            else:
                data[i] = values[0]
            print('\r%d/%d' % (i,size[0]), end='')
        print('\r%d/%d'% (size[0], size[0]))
    return data

class mnist_dataset():
    def __init__(self, train=True, download=True, frac=1., oneHot=True, transform=None):
        if train:
            data_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
            label_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        else:
            data_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
            label_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

        if download:
            if not os.path.exists(data_url[data_url.rfind('/')+1:]):
                with open(data_url[data_url.rfind('/')+1:],'wb') as fout:
                    r = requests.get(data_url)
                    fout.write(r.content)

            if not os.path.exists(label_url[label_url.rfind('/')+1:]):
                with open(label_url[label_url.rfind('/')+1:],'wb') as fout:
                    r = requests.get(label_url)
                    fout.write(r.content)

        print('Loading %s images...' % ('training' if train else 'testing'))
        self._images = read_idx(data_url[data_url.rfind('/')+1:])
        print('Loading %s labels...' % ('training' if train else 'testing'))
        self._labels = read_idx(label_url[label_url.rfind('/')+1:])

        self._oneHot = oneHot
        self._tform = transform

        shuffle = np.arange(self._images.shape[0])
        np.random.shuffle(shuffle)
        self._images = self._images[shuffle[:int(self._images.shape[0]*frac)]]
        self._labels = self._labels[shuffle[:int(self._labels.shape[0]*frac)]]

    def __len__(self):
        return(self._labels.shape[0])

    def __getitem__(self, idx):
        if idx >= self._labels.shape[0]:
            raise IndexError('Index %d, greater than dataset size %d' % (idx, self._labels.shape[0]))

        img = self._images[idx]
        if self._tform is not None:
            img = self._tform(img)

        if self._oneHot:
            lbl = np.zeros((10,))
            lbl[self._labels[idx]] = 1
        else:
            lbl = self._labels[idx:idx+1] #index as a range len 1 to ensure lbl is a numpy array
        label = coatl.tensor(data=lbl)

        return img, label