import logging

import numpy as np
from sklearn.model_selection import train_test_split

from utils import logging as lg


lg.set_logging()


def get_mnist(dataset, dir_path='./data/mnist'):

    if dataset == 'train':
        prefix = 'train'
    elif dataset == 'test':
        prefix = 't10k'
    else:
        raise ValueError('No dataset MNIST - %s' % dataset)

    logging.debug('Load MNIST : %s' % dataset)

    x_path = '%s/%s-images-idx3-ubyte' % (dir_path, prefix)
    y_path = '%s/%s-labels-idx1-ubyte' % (dir_path, prefix)

    with open(x_path) as xf:
        with open(y_path) as yf:
            x = 2.0*np.fromfile(xf, dtype='ubyte', count=-1)[16:].reshape((-1, 784)) / 255 - 1
            y = np.fromfile(yf, dtype='ubyte', count=-1)[8:]
            y = (y[:, np.newaxis] == np.arange(10)) * 1.0
    return x, y


def get_empty_data():
    return np.zeros((28, 28)) - 1


class DataSet:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_batch(self, no_batch):
        total = len(self.x)
        for ndx in range(0, total, no_batch):
            yield (self.x[ndx:min(ndx + no_batch, total)], self.y[ndx:min(ndx + no_batch, total)])


class MNISTData:
    def __init__(self, dir_path='./data/mnist'):

        x_train, y_train = get_mnist('train', dir_path=dir_path)
        x_test, y_test = get_mnist('test', dir_path=dir_path)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=71)

        self.train = DataSet(x_train, y_train)
        self.val = DataSet(x_val, y_val)
        self.test = DataSet(x_test, y_test)

        self.train2d = DataSet(x_train.reshape(-1, 28, 28), y_train)
        self.val2d = DataSet(x_val.reshape(-1, 28, 28), y_val)
        self.test2d = DataSet(x_test.reshape(-1, 28, 28), y_test)
