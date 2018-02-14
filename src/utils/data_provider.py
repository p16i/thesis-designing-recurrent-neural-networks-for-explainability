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
    elif data == 'mnist-3-digits':
        return MNIST3DigitsData()
    elif data == 'mnist-3-digits-maj':
        return MNIST3DigitsWithMajorityData()
    elif data == 'fashion-mnist':
        return FashionMNISTData()
    elif data == 'fashion-mnist-3-items':
        return FashionMNIST3ItemsData()
    elif data == 'fashion-mnist-3-items-maj':
        return FashionMNIST3DigitsWithMajorityData()
    elif data == 'ufi-cropped':
        return UFICroppedData()


def fill_left_right_digit(x, y, seed=71):
    new_x = np.zeros((x.shape[0], 28, 28*3))

    new_x[:, :, 28:(28*2)] = x

    def plot_sample_propotions(indices, label):
        y_not_in_class_i = classes[indices]
        counts = dict()
        for jj in range(10):
            counts[jj] = np.sum(y_not_in_class_i == jj)

        logging.info('%s | sample propotions' % (label))
        logging.info(counts)

    np.random.seed(seed)
    classes = np.argmax(y, axis=1)

    # plot_sample_propotions(range(classes.shape[0]), 'total')

    for i in range(10):
        samples_in_class_i = np.squeeze(np.argwhere(classes == i))
        total = samples_in_class_i.shape[0]

        samples_not_in_class_i = np.squeeze(np.argwhere(classes != i))

        left_indices = np.random.choice(samples_not_in_class_i, total)
        # plot_sample_propotions(left_indices, 'left-%d' % i)

        right_indices = np.random.choice(samples_not_in_class_i, total)
        # plot_sample_propotions(right_indices, 'right-%d' % i)

        new_x[samples_in_class_i, :, :28] = x[left_indices, :, :]
        new_x[samples_in_class_i, :, -28:] = x[right_indices, :, :]

    return new_x, y


def create_majority_data(x, y, seed=71):
    np.random.seed(seed)
    classes = np.argmax(y, axis=1)

    new_x = np.tile(x, (1, 3))

    fake_digit_positions = np.zeros((new_x.shape[0], 3))

    np.random.choice([0, 1, 2], new_x.shape[0], replace=True)

    for i in range(10):
        samples_in_class_i = np.squeeze(np.argwhere(classes == i))
        total = samples_in_class_i.shape[0]

        samples_not_in_class_i = np.squeeze(np.argwhere(classes != i))

        fake_digit_idx = np.random.choice(samples_not_in_class_i, total)
        same_class_digit_idx = np.random.choice(samples_in_class_i, total)

        for j, idx in zip(range(total), samples_in_class_i):
            dd = [x[idx, :, :], x[same_class_digit_idx[j], :, :], x[fake_digit_idx[j], :, :]]
            permuted_pos = np.random.permutation(range(3))

            fake_digit_positions[idx] = permuted_pos
            dd_permuted = [dd[jj] for jj in permuted_pos]

            new_x[idx, :, :] = np.concatenate(dd_permuted, axis=1)

    return new_x, y, fake_digit_positions


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

        self.labels = {
            0: 'Digit 0',
            1: 'Digit 1',
            2: 'Digit 2',
            3: 'Digit 3',
            4: 'Digit 4',
            5: 'Digit 5',
            6: 'Digit 6',
            7: 'Digit 7',
            8: 'Digit 8',
            9: 'Digit 9'
        }

    def get_text_label(self, label_index):
        return 'Digit %d' % label_index

    def get_samples_for_vis(self, n=12):

        np.random.seed(1234)

        r = np.random.randint(0,  self.test2d.y.shape[0], n)

        return self.test2d.x[r, :, :], self.test2d.y[r]


class MNIST3DigitsData(MNISTData):
    def __init__(self, **kwargs):
        super(MNIST3DigitsData, self).__init__(**kwargs)

        self.dims = (28, 28*3)

        self.train2d = DataSet(*fill_left_right_digit(self.train2d.x, self.train2d.y, seed=0))
        self.val2d = DataSet(*fill_left_right_digit(self.val2d.x, self.val2d.y, seed=1))
        self.test2d = DataSet(*fill_left_right_digit(self.test2d.x, self.test2d.y, seed=3))

        self.train = self.train2d
        self.val = self.val2d
        self.test = self.test2d


class MNIST3DigitsWithMajorityData(MNISTData):
    def __init__(self, **kwargs):
        super(MNIST3DigitsWithMajorityData, self).__init__(**kwargs)

        self.dims = (28, 28*3)

        x, y, fake_digit_position = create_majority_data(self.train2d.x, self.train2d.y, seed=0)
        self.train2d = DataSet(x, y)
        self.train2d_fake_digit_position = fake_digit_position
        assert self.train2d_fake_digit_position.shape[0] == self.train2d.y.shape[0]

        x, y, fake_digit_position = create_majority_data(self.val2d.x, self.val2d.y, seed=1)
        self.val2d = DataSet(x, y)
        self.val2d_fake_digit_position = fake_digit_position
        assert self.val2d_fake_digit_position.shape[0] == self.val2d.y.shape[0]

        x, y, fake_digit_position = create_majority_data(self.test2d.x, self.test2d.y, seed=3)
        self.test2d = DataSet(x, y)
        self.test2d_fake_digit_position = fake_digit_position
        assert self.test2d_fake_digit_position.shape[0] == self.test2d.y.shape[0]

        self.train = self.train2d
        self.val = self.val2d
        self.test = self.test2d


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


class FashionMNIST3ItemsData(FashionMNISTData):
    def __init__(self, **kwargs):
        super(FashionMNIST3ItemsData, self).__init__(**kwargs)

        self.dims = (28, 28*3)

        self.train2d = DataSet(*fill_left_right_digit(self.train2d.x, self.train2d.y, seed=20))
        self.val2d = DataSet(*fill_left_right_digit(self.val2d.x, self.val2d.y, seed=21))
        self.test2d = DataSet(*fill_left_right_digit(self.test2d.x, self.test2d.y, seed=23))

        self.train = self.train2d
        self.val = self.val2d
        self.test = self.test2d


class FashionMNIST3DigitsWithMajorityData(FashionMNISTData):
    def __init__(self, **kwargs):
        super(FashionMNIST3DigitsWithMajorityData, self).__init__(**kwargs)

        self.dims = (28, 28*3)

        x, y, fake_digit_position = create_majority_data(self.train2d.x, self.train2d.y, seed=0)
        self.train2d = DataSet(x, y)
        self.train2d_fake_digit_position = fake_digit_position
        assert self.train2d_fake_digit_position.shape[0] == self.train2d.y.shape[0]

        x, y, fake_digit_position = create_majority_data(self.val2d.x, self.val2d.y, seed=1)
        self.val2d = DataSet(x, y)
        self.val2d_fake_digit_position = fake_digit_position
        assert self.val2d_fake_digit_position.shape[0] == self.val2d.y.shape[0]

        x, y, fake_digit_position = create_majority_data(self.test2d.x, self.test2d.y, seed=3)
        self.test2d = DataSet(x, y)
        self.test2d_fake_digit_position = fake_digit_position
        assert self.test2d_fake_digit_position.shape[0] == self.test2d.y.shape[0]

        self.train = self.train2d
        self.val = self.val2d
        self.test = self.test2d


class UFICroppedData:
    def __init__(self, dir_path='./data/ufi-cropped'):
        # subsampling_indices = list(np.arange(0, 128, 2))

        def avg_pooling(x):
            new_x = np.zeros((x.shape[0], 64, 64))

            for i in range(0, x.shape[1], 2):
                for j in range(0, x.shape[2], 2):
                    new_x[:, int(i/2), int(j/2)] = np.mean(x[:, i:(i+2), j:(j+2)].reshape(-1, 4), axis=1)

            return new_x

        def flip_data(x, y):
            total = x.shape[0]
            new_x = np.tile(x, (2, 1, 1))
            new_y = np.tile(y, (2, 1))

            new_x[total:, :, :] = x[:, :, ::-1]

            np.random.seed(0)

            shuffled_indices = np.arange(total*2)
            np.random.shuffle(shuffled_indices)

            return new_x[shuffled_indices, :, :], new_y[shuffled_indices, :]

        x_train = avg_pooling(np.load('%s/train-x.npy' % dir_path).reshape(-1, 128, 128))
        y_train = np.load('%s/train-y.npy' % dir_path)

        # print(x_train[0])
        # print(np.argmax(y_train[0]))

        x_train, y_train = flip_data(x_train, y_train)
        # print('Train data', x_train.shape)

        x_test = avg_pooling(np.load('%s/test-x.npy' % dir_path).reshape(-1, 128, 128))
        y_test = np.load('%s/test-y.npy' % dir_path)

        x_test, y_test = flip_data(x_test, y_test)
        # print('Test data', x_test.shape)

        self.dims = (64, 64)

        # This is a bad idea but we have limited amount of data
        x_val, y_val = x_test, y_test

        self.no_classes = y_test.shape[1]

        self.train = DataSet(x_train, y_train)
        self.val = DataSet(x_val, y_val)
        self.test = DataSet(x_test, y_test)

        self.train2d = DataSet(x_train, y_train)
        self.val2d = DataSet(x_val, y_val)
        self.test2d = DataSet(x_test, y_test)

    def get_samples_for_vis(self, n=12):

        print("WARNING! this is data sampled from training set not testing one")
        indices = [2785, 2973, 57, 906, 393, 3666, 3502, 1222, 731, 2659, 3400, 656]

        return self.train2d.x[indices, :], self.train2d.y[indices]

    def get_text_label(self, label_index):
        return 'Person %d' % label_index
