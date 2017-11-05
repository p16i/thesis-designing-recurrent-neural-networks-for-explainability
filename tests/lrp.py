import unittest
import logging
import tensorflow as tf

from utils import data_provider
from model import provider

tf.logging.set_verbosity(tf.logging.ERROR)
logging.disable(logging.DEBUG)

NO_TESTING_DATA = 5


class TestLRP(unittest.TestCase):
    def test_s2(self):
        TestLRP._test_lrp('../tests/data/s2-network')

    def test_s3(self):
        TestLRP._test_lrp('../tests/data/s3-network')

    @staticmethod
    def _test_lrp(model):
        data = data_provider.MNISTData(dir_path='../data/mnist')
        model = provider.load(model)

        model.lwr(data.test2d.x[:NO_TESTING_DATA, :, :], debug=True)


if __name__ == '__main__':
    unittest.main()
