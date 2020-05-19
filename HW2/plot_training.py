import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re

files = os.listdir()
files = [f for f in files if re.match('.*\.txt', f)]
print(files)

for fname in files:
    train_loss = np.zeros((0,))
    test_loss = np.zeros((0,))
    test_acc = np.zeros((0,))
    train_time = 0

    with open(fname, 'r') as fin:
        for line in fin:
            if re.match('^Training.*', line):
                m = re.search('(?<=total loss: )[0-9]+\.[0-9]+', line)
                if m:
                    train_loss = np.append(train_loss, [float(m.group(0))])
            elif re.match('^Testing.*', line):
                m = re.search('(?<=test loss: )[0-9]+\.[0-9]+', line)
                if m:
                    test_loss = np.append(test_loss, [float(m.group(0))])
                m = re.search('(?<=accuracy: )[0-9]+\.[0-9]+', line)
                if m:
                    test_acc = np.append(test_acc, [float(m.group(0))])
            elif re.match('^Total.*', line):
                m = re.search('(?<=Total Training Time: )[0-9]+\.[0-9]+', line)
                if m:
                    train_time = float(m.group(0))
    t = np.arange(0,5,5/train_loss.shape[0])
    plt.plot(t, train_loss, label='Training Loss')
    plt.plot(np.arange(1,6), test_loss, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if re.match('^relu_k[0-9]+.*', fname):
        m = re.search('[0-9]+(?=\.txt)', fname)
        plt.title('ReLU Training k=%d' % int(m.group(0)))
    elif re.match('sigmoid_k[0-9]+.*', fname):
        m = re.search('[0-9]+(?=\.txt)', fname)
        plt.title('Sigmoid Training k=%d' % int(m.group(0)))
    elif re.match('leaky_relu_k[0-9]+.*', fname):
        m = re.search('[0-9]+(?=\.txt)', fname)
        plt.title('Leaky ReLU Training k=%d' % int(m.group(0)))
    else:
        plt.title('Linear Training')
    plt.legend()
    plt.show()

    plt.plot(np.arange(1,6), test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    if re.match('^relu_k[0-9]+.*', fname):
        m = re.search('[0-9]+(?=\.txt)', fname)
        plt.title('ReLU Testing k=%d' % int(m.group(0)))
    elif re.match('sigmoid_k[0-9]+.*', fname):
        m = re.search('[0-9]+(?=\.txt)', fname)
        plt.title('Sigmoid Testing k=%d' % int(m.group(0)))
    elif re.match('leaky_relu_k[0-9]+.*', fname):
        m = re.search('[0-9]+(?=\.txt)', fname)
        plt.title('Leaky ReLU Testing k=%d' % int(m.group(0)))
    else:
        plt.title('Linear Testing')
    plt.legend()
    plt.show()