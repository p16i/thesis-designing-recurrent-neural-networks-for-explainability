import unittest

from utils import network_architecture


class TestNetworkArchitecture(unittest.TestCase):
    def test_parse(self):
        res = network_architecture.parse('in1:200|in2:100|hidden:30|out1:30|out2:10--recur:10')
        exp = dict(in1=200, in2=100, hidden=30, out1=30, out2=10, recur=10)
        self.assertDictEqual(res, exp)

        s = 'conv1:5x5x32=>3x4[5,6]|conv2:5x5x64=>2x2[2,2]|hidden:30|out1:30|out2:10--recur:10'
        res = network_architecture.parse(s)
        exp = dict(
            conv1=dict(
                conv={
                    'kernel_size': [5, 5],
                    'filters': 32,
                },
                pooling={
                    'kernel_size': [3, 4],
                    'strides': [5, 6]
                }
            ),
            conv2=dict(
                conv={
                    'kernel_size': [5, 5],
                    'filters': 64,
                },
                pooling={
                    'kernel_size': [2, 2],
                    'strides': [2, 2]
                }
            ),
            hidden=30,
            out1=30,
            out2=10,
            recur=10
        )
        self.assertDictEqual(res, exp)


if __name__ == '__main__':
    unittest.main()