import unittest

from dummy_module import foo


class TestFoo(unittest.TestCase):
    def test_foo(self):
        res = foo('something')
        self.assertEqual('foo something', res)

    def test_falied_foo(self):
        res = foo('something')
        self.assertEqual('fx something', res)


if __name__ == '__main__':
    unittest.main()