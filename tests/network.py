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


class TestNetwork(unittest.TestCase):
    def test_s2(self):
        TestNetwork._test_lrp('tests/data/s2-network')

    def test_s3(self):
        TestNetwork._test_lrp('tests/data/s3-network')

    def test_deep_4l(self):
        TestNetwork._test_lrp('tests/data/deep-4l-network')

    def test_convdeep_4l(self):
        TestNetwork._test_lrp('tests/data/convdeep-4l-network')

    def test_no_variables(self):
        networks = [('s2-network', 325386), ('s3-network', 198218)]
        for network, expected in networks:
            model_obj = TestNetwork._load_model('tests/data/%s' % network)
            no_variables = model_obj.dag.no_variables()

            print(model_obj.experiment_artifact)
            self.assertEqual(int(no_variables), expected, 'Correct no. variables of %s' % network)

    @staticmethod
    def _test_lrp(model):
        data = data_provider.MNISTData(dir_path=prepend_project_root('/data/mnist'))
        model = TestNetwork._load_model(model)

        model.lrp(data.test2d.x[:NO_TESTING_DATA, :, :], debug=True)

    @staticmethod
    def _load_model(model):
        return provider.load(prepend_project_root(model))


if __name__ == '__main__':
    unittest.main()
