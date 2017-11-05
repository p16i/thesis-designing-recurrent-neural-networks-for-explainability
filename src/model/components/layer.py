import tensorflow as tf
from utils import logging as lg

lg.set_logging()


class Layer:
    def __init__(self, dims, name, stddev=0.1):
        weights = tf.Variable(
            tf.truncated_normal(dims, stddev=stddev),
            name="%s_weights" % name
        )

        bias = tf.Variable(
            tf.zeros(dims[1]),
            name="%s_bias" % name
        )

        self.W = weights
        self.b = bias

    def get_no_variables(self):
        return self.W.shape[0] * self.W.shape[1] + self.b.shape[0]
