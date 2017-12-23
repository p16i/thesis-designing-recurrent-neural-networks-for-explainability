import logging
import numpy as np
import tensorflow as tf
from utils import logging as lg

lg.set_logging()

DIVISION_ADJUSTMENT=1e-9
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

    def rel_z_plus_prop(self, x, relevance, factor=1.0):
        v = tf.maximum(0.0, self.W) + (1.0 - factor) * tf.minimum(0.0, self.W)

        z = tf.matmul(x, v) + DIVISION_ADJUSTMENT
        s = relevance / z
        c = tf.matmul(s, tf.transpose(v))
        return x * c

    def rel_z_beta_prop(self, x, relevance, lowest=-1.0, highest=1.0, factor=1):
        w, v, u = self.W, tf.maximum(0.0, self.W), tf.minimum(0.0, self.W)
        l, h = x * 0 + lowest, x * 0 + highest

        z = tf.matmul(x, w) - factor*(tf.matmul(l, v) + tf.matmul(h, u)) + DIVISION_ADJUSTMENT
        s = relevance / z
        return x * tf.matmul(s, tf.transpose(w))\
               - factor*(l * tf.matmul(s, tf.transpose(v)) + h * tf.matmul(s, tf.transpose(u)))

    @staticmethod
    def rel_z_plus_beta_prop(x_p, w_p, x_b, w_b, relevance, factor=1.0, lowest=-1, highest=1):
        v_p = tf.maximum(0.0, w_p) + (1.0 - factor) * tf.minimum(0.0, w_p)
        z_p = tf.matmul(x_p, v_p) + DIVISION_ADJUSTMENT

        w_b, v_b, u_b = w_b, tf.maximum(0.0, w_b), tf.minimum(0.0, w_b)
        l_b, h_b = x_b * 0 + lowest, x_b * 0 + highest
        z_b = tf.matmul(x_b, w_b) - factor*(tf.matmul(l_b, v_b) + tf.matmul(h_b, u_b)) + DIVISION_ADJUSTMENT

        z = z_p + z_b

        s = relevance / z

        # z-plus
        c_p = tf.matmul(s, tf.transpose(v_p))
        r_p = x_p * c_p

        # z-beta
        r_b = x_b * tf.matmul(s, tf.transpose(w_b)) \
              - factor*(l_b * tf.matmul(s, tf.transpose(v_b)) + h_b * tf.matmul(s, tf.transpose(u_b)))

        return r_p, r_b


class ConvolutionalLayer(Layer):
    def __init__(self, input_channels, kernel_size, filters, name, default_weights=None, default_biases=None):
        super().__init__(kernel_size + [input_channels, filters], name,
                                                 default_weights=default_weights, default_biases=default_biases)

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = [1]*4
        self.padding = 'SAME'

    def clone(self):
        c = ConvolutionalLayer(self.input_channels, self.kernel_size, self.filters, '%s-copy' % self.name,
                               self.W, self.b)

        return c

    def conv_with_w(self, x, w):
        return tf.nn.conv2d(x, w, strides=self.strides, padding=self.padding)

    def conv(self, x):
        hconv = self.conv_with_w(x, self.W)
        hconv_relu = tf.nn.relu(hconv - tf.nn.softplus(self.b))

        return hconv, hconv_relu

    def rel_zplus_prop(self, x, relevance, factor=1):
        v = tf.maximum(0.0, self.W) + (1.0 - factor) * tf.minimum(0.0, self.W)

        hconv = self.conv_with_w(x, v)

        z = hconv + DIVISION_ADJUSTMENT
        s = relevance / z

        c = tf.nn.conv2d_backprop_input(
            tf.shape(x), v,
            out_backprop=s,
            strides=self.strides,
            padding=self.padding
        )

        return x*c

    def rel_zbeta_prop(self, x, relevance, lowest=-1, highest=1, factor=1):

        w_neg = tf.minimum(0.0, self.W)
        w_pos = tf.maximum(0.0, self.W)

        l, h = x*0.0+lowest, x*0+highest

        i_act, _ = self.conv(x)
        p_act = self.conv_with_w(l, w_pos)
        n_act = self.conv_with_w(h, w_neg)

        s = relevance / (i_act - factor*(p_act + n_act) + DIVISION_ADJUSTMENT)

        shape_x = tf.shape(x)

        grad_params = dict(
            out_backprop=s,
            strides=self.strides,
            padding=self.padding
        )

        R = x*tf.nn.conv2d_backprop_input(shape_x, self.W, **grad_params) - \
            l*tf.nn.conv2d_backprop_input(shape_x, w_pos, **grad_params) - \
            h*tf.nn.conv2d_backprop_input(shape_x, w_neg, **grad_params) \

        return R


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

    def rel_prop(self, x, activations, relevance):
        s = relevance / (activations + DIVISION_ADJUSTMENT)
        c = tf.gradients(activations, x, grad_ys=s)[0]

        return x*c
