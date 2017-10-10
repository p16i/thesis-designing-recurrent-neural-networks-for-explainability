import unittest

from utils import network_architecture

class TestFoo(unittest.TestCase):
    def test_foo(self):
        res = network_architecture.parse_architecture('in1:200|in2:100|hidden:30|out1:30|out2:10--recur:10')
        exp = dict(in1=200, in2=100, hidden=30, out1=30, out2=10, recur=10)
        self.assertDictEqual(res, exp)


if __name__ == '__main__':
    unittest.main()