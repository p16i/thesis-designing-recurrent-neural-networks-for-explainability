import unittest
import logging
import tensorflow as tf
import os

from utils import data_provider
from model import provider

tf.logging.set_verbosity(tf.logging.ERROR)
logging.disable(logging.DEBUG)

NO_TESTING_DATA = 5

PROJECT_ROOT = '/'.join(os.path.abspath(__file__).split('/')[:-2])


def prepend_project_root(path):
    return '%s/%s' % (PROJECT_ROOT, path)


class TestLRP(unittest.TestCase):
    def test_s2(self):
        TestLRP._test_lrp(prepend_project_root('tests/data/s2-network'))

    def test_s3(self):
        TestLRP._test_lrp(prepend_project_root('tests/data/s3-network'))

    @staticmethod
    def _test_lrp(model):
        data = data_provider.MNISTData(dir_path= prepend_project_root('/data/mnist'))
        model = provider.load(model)

        model.lrp(data.test2d.x[:NO_TESTING_DATA, :, :], debug=True)


if __name__ == '__main__':
    unittest.main()
