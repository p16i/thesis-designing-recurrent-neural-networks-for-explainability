import logging

import numpy as np

from utils import logging as lg

from prettytable import PrettyTable

lg.set_logging()


def get_mnist(dataset, dir_path='./data/mnist'):

    if dataset == 'train':
        prefix = 'train'
    elif dataset == 'test':
        prefix = 't10k'
    else:
        raise ValueError('No dataset MNIST - %s' % dataset)

    logging.info('Load MNIST : %s' % dataset)

    x_path = '%s/%s-images-idx3-ubyte' % (dir_path, prefix)
    y_path = '%s/%s-labels-idx1-ubyte' % (dir_path, prefix)

    with open(x_path) as xf:
        with open(y_path) as yf:
            x = np.fromfile(xf, dtype='ubyte', count=-1)[16:].reshape((-1, 784)) / 255
            y = np.fromfile(yf, dtype='ubyte', count=-1)[8:]
            y = (y[:, np.newaxis] == np.arange(10)) * 1.0
    return x, y


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

        self.train = DataSet(x_train, y_train)
        self.test = DataSet(x_test, y_test)

        self.train2d = DataSet(x_train.reshape(-1, 28, 28), y_train)
        self.test2d = DataSet(x_test.reshape(-1, 28, 28), y_test)
