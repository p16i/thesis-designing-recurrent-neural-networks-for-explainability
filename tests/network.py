import unittest
import logging
import tensorflow as tf
import os

from utils import data_provider
from model import provider

tf.logging.set_verbosity(tf.logging.ERROR)
logging.disable(logging.DEBUG)

NO_TESTING_DATA = 100

PROJECT_ROOT = '/'.join(os.path.abspath(__file__).split('/')[:-2])


def prepend_project_root(path):
    return '%s/%s' % (PROJECT_ROOT, path)


class TestNetwork(unittest.TestCase):
    def test_s2(self):
        TestNetwork._test_lrp('final-models/s2_network-mnist-seq-7')

    def test_s3(self):
        TestNetwork._test_lrp('final-models/s3_network-mnist-seq-7')

    def test_deep_4l(self):
        TestNetwork._test_lrp('final-models/deep_4l_network-mnist-seq-7')

    def test_convdeep_4l(self):
        TestNetwork._test_lrp('final-models/convdeep_4l_network-mnist-seq-7')

    def test_shallow_2level(self):
        TestNetwork._test_lrp('experiment-results/shallow_2_levels/shallow_2_levels-mnist-seq-7---2018-02-11--21-05-33')

    def test_no_variables(self):
        networks = [('final-models/s2_network-mnist-seq-7', 291210),
                    ('final-models/s3_network-mnist-seq-7', 271946)]
        for network, expected in networks:
            model_obj = TestNetwork._load_model(network)
            no_variables = model_obj.dag.no_variables()

            print(model_obj.experiment_artifact)
            self.assertEqual(int(no_variables), expected, 'Correct no. variables of %s' % network)

    @staticmethod
    def _test_lrp(model):
        data = data_provider.MNISTData(dir_path=prepend_project_root('/data/mnist'))
        model = TestNetwork._load_model(model)

        start_idx = 100
        x = data.test2d.x[start_idx:(start_idx+NO_TESTING_DATA), :, :]
        y = data.test2d.y[start_idx:(start_idx+NO_TESTING_DATA), :]
        model.rel_lrp_deep_taylor(x, y, debug=True)
        # model.rel_lrp_alpha2_beta1(x, y, debug=True)

    @staticmethod
    def _load_model(model):
        return provider.load(prepend_project_root(model))


if __name__ == '__main__':
    unittest.main()
