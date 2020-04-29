import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re

files = ['pytorch_training.txt']
#files = os.listdir()
#files = [f for f in files if re.match('.*\.txt', f)]
#print(files)

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
    t = np.arange(0,10,10/train_loss.shape[0])
    plt.plot(t, train_loss, label='Training Loss')
    plt.plot(np.arange(1,11), test_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if re.match('^exp_bs.*', fname):
        m = re.search('[0-9]+(?=\.txt)', fname)
        plt.title('Batch Size: %d' % int(m.group(0)))
    elif re.match('^exp_df.*', fname):
        m = re.search('[0-9]+(?=\.txt)', fname)
        plt.title('Training Set Size: %d' % int(m.group(0)))
    else:
        plt.title('Pytorch Training')
    plt.legend()
    plt.show()

    plt.plot(np.arange(1,11), test_acc, label='Test Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Loss')
    if re.match('^exp_bs.*', fname):
        m = re.search('[0-9]+(?=\.txt)', fname)
        plt.title('Batch Size: %d' % int(m.group(0)))
    elif re.match('^exp_df.*', fname):
        m = re.search('[0-9]+(?=\.txt)', fname)
        plt.title('Training Set Size: %d' % int(m.group(0)))
    else:
        plt.title('Pytorch Testing')
    plt.legend()
    plt.show()