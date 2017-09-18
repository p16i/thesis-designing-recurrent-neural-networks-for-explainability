import unittest

from dummy_module.foo import foo


class TestFoo(unittest.TestCase):
    def test_foo(self):
        res = foo('something')
        self.assertEqual('foo something', res)


if __name__ == '__main__':
    unittest.main()