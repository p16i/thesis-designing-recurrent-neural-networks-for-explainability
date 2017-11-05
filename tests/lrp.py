import unittest
import logging

from utils import data_provider

from model import provider

logging.disable(logging.DEBUG)


NO_TESTING_DATA = 5


class TestLRP(unittest.TestCase):
    def test_s2_network_lrp(self):
        data = data_provider.MNISTData(dir_path='../data/mnist')
        model = provider.load('../tests/data/s2-network')

        model.lwr(data.test2d.x[:NO_TESTING_DATA, :, :], debug=True)


if __name__ == '__main__':
    unittest.main()
