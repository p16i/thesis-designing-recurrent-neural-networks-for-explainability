import logging
import numpy as np
import tensorflow as tf
from utils import logging as lg

lg.set_logging()

DEFAULT_BIAS_VALUE=0.01

class Layer:
    def __init__(self, dims, name, stddev=0.1, default_weights=None, default_biases=None):

        w_name = "%s_weights" % name
        b_name = "%s_bias" % name

        if default_weights is None:
            self.W = tf.Variable(tf.truncated_normal(dims, stddev=stddev), name=w_name)
        else:
            logging.info('Set default weights manually for layer %s' % name)
            self.W = tf.identity(default_weights, name=w_name)

        if default_biases is None:
            # this make bias after softmax is 0.01
            self.b = tf.constant( float(np.log(np.exp(DEFAULT_BIAS_VALUE)-1)), shape=dims[-1:], name=b_name)
        else:
            logging.info('Set default biases manually for layer %s' % name)
            self.b = tf.identity(default_biases, name=b_name)

        self.name = name

    def get_no_variables(self):
        return int(np.prod(self.W.shape) + self.b.shape[0])


class ConvolutionalLayer(Layer):
    def __init__(self, input_channels, kernel_size, filters, name, default_weights=None, default_biases=None):
        super().__init__(kernel_size + [input_channels, filters], name,
                                                 default_weights=default_weights, default_biases=default_biases)

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = [1]*4
        self.padding = 'SAME'

    def conv(self, inputs):
        hconv = tf.nn.conv2d(inputs, self.W, strides=self.strides, padding=self.padding)
        hconv_relu = tf.nn.relu(hconv - tf.nn.softplus(self.b))

        return hconv, hconv_relu

    def clone(self):
        c = ConvolutionalLayer(self.input_channels, self.kernel_size, self.filters, '%s-copy' % self.name,
                               self.W, self.b)

        return c


class PoolingLayer:
    def __init__(self, kernel_size, strides):
        self.kernel_size = kernel_size
        self.strides = strides

    def pool(self, x):
        return tf.nn.max_pool(
            x,
            ksize=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1], padding='SAME'
        )

    def get_no_variables(self):
        return 0

