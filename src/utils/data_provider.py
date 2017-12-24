import logging

import numpy as np
from sklearn.model_selection import train_test_split

from utils import logging as lg
from heatmap_tutorial import utils as ht_utils


lg.set_logging()


def get_mnist(dataset, dir_path='./data/mnist'):

    if dataset == 'train':
        prefix = 'train'
    elif dataset == 'test':
        prefix = 't10k'
    else:
        raise ValueError('No dataset MNIST - %s' % dataset)

    logging.debug('Load %s : %s' % (dir_path, dataset))

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


def get_data(data):
    if data == 'mnist':
        return MNISTData()
    elif data == 'fashion-mnist':
        return FashionMNISTData()
    elif data == 'ufi-cropped':
        return UFICroppedData()


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

        self.dir_path = dir_path

        x_train, y_train = get_mnist('train', dir_path=dir_path)
        x_test, y_test = get_mnist('test', dir_path=dir_path)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=71)

        self.no_classes = 10
        self.dims = (28, 28)

        self.train = DataSet(x_train, y_train)
        self.val = DataSet(x_val, y_val)
        self.test = DataSet(x_test, y_test)

        self.train2d = DataSet(x_train.reshape(-1, 28, 28), y_train)
        self.val2d = DataSet(x_val.reshape(-1, 28, 28), y_val)
        self.test2d = DataSet(x_test.reshape(-1, 28, 28), y_test)

    def get_text_label(self, label_index):
        return 'Digit %d' % label_index

    def get_samples_for_vis(self, n=12):

        x, y = ht_utils.getMNISTsample(n, path=self.dir_path, seed=1234)

        return x.reshape(-1, 28, 28), y

class FashionMNISTData:
    def __init__(self, dir_path='./data/fashion-mnist'):

        x_train, y_train = get_mnist('train', dir_path=dir_path)
        x_test, y_test = get_mnist('test', dir_path=dir_path)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=71)

        self.dims = (28, 28)
        self.no_classes = 10

        self.train = DataSet(x_train, y_train)
        self.val = DataSet(x_val, y_val)
        self.test = DataSet(x_test, y_test)

        self.train2d = DataSet(x_train.reshape(-1, 28, 28), y_train)
        self.val2d = DataSet(x_val.reshape(-1, 28, 28), y_val)
        self.test2d = DataSet(x_test.reshape(-1, 28, 28), y_test)

        self.labels = {
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        }

    def get_samples_for_vis(self, n=12):

        indices = [588, 314, 47, 145, 258, 641, 561, 3410, 1094, 4059, 518, 9304]

        return self.test2d.x[indices, :], self.test2d.y[indices]

    def get_text_label(self, label_index):
        return self.labels[label_index]


class UFICroppedData:
    def __init__(self, dir_path='./data/ufi-cropped'):
        x_train = np.load('%s/train-x.npy' % dir_path)
        y_train = np.load('%s/train-y.npy' % dir_path)

        x_test = np.load('%s/test-x.npy' % dir_path)
        y_test = np.load('%s/test-y.npy' % dir_path)

        # This is a bad idea but we have limited amount of data
        x_val, y_val = x_test, y_test

        self.dims = (128, 128)
        self.no_classes = 605

        self.train = DataSet(x_train, y_train)
        self.val = DataSet(x_val, y_val)
        self.test = DataSet(x_test, y_test)

        self.train2d = DataSet(x_train.reshape(-1, self.dims[0], self.dims[1]), y_train)
        self.val2d = DataSet(x_val.reshape(-1, self.dims[0], self.dims[1]), y_val)
        self.test2d = DataSet(x_test.reshape(-1, self.dims[0], self.dims[1]), y_test)

    def get_samples_for_vis(self, n=12):

        indices = [2785, 2973, 57, 906, 393, 3666, 3502, 1222, 731, 2659, 3400, 656]

        return self.test2d.x[indices, :], self.test2d.y[indices]

    def get_text_label(self, label_index):
        return 'Person %d' % label_index
