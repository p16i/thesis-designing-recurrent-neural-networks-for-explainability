import logging
import numpy as np
import tensorflow as tf
from utils import logging as lg

lg.set_logging()

DIVISION_ADJUSTMENT=1e-9
DEFAULT_BIAS_VALUE=0.0

class Layer:
    def __init__(self, dims, name, default_weights=None, default_biases=None, no_input_activations=None):

        w_name = "%s_weights" % name
        b_name = "%s_bias" % name

        if no_input_activations is None and len(dims) == 2:
            if len(dims) == 2:
                no_input_activations = dims[0]
                logging.info('setting no_input_activations to %d' % no_input_activations)
            else:
                logging.error('no_input_activations should be specified when dims`s range > 2')

        logging.info("Layer %s with %d input activations" % (name,no_input_activations))

        if default_weights is None:
            self.W = tf.Variable(tf.truncated_normal(dims, stddev=1.0/np.sqrt(no_input_activations)), name=w_name)
        else:
            logging.info('Set default weights manually for layer %s' % name)
            self.W = tf.identity(default_weights, name=w_name)

        if default_biases is None:
            # this make bias after softmax is 0.01
            self.b = tf.Variable(tf.ones(dims[-1]) * float(np.log(np.exp(DEFAULT_BIAS_VALUE)-1)), name=b_name)
        else:
            logging.info('Set default biases manually for layer %s' % name)
            self.b = tf.identity(default_biases, name=b_name)

        self.name = name

    def get_no_variables(self):
        return int(np.prod(self.W.shape) + self.b.shape[0])

    def rel_z_plus_prop(self, x, relevance, alpha, beta):
        wp = tf.maximum(0.0, self.W)
        wn = tf.minimum(0.0, self.W)

        def compute_c(w):
            z = tf.matmul(x, w)
            s = relevance / (z + DIVISION_ADJUSTMENT)
            return tf.matmul(s, tf.transpose(w))

        return x * (alpha*compute_c(wp) - beta*compute_c(wn))

    def rel_lrp_for_lstm(self, xin, hout, relevance, delta=1, eps=0.001):
        eps_sign = eps*tf.where(hout >= 0, tf.ones(tf.shape(hout)), -tf.ones(tf.shape(hout)))  # shape (1, M)

        hout_adj = hout + eps_sign

        relevance_adj = relevance / hout_adj

        number_bias_unit = self.W.get_shape()[0].value
        bias_term = (eps_sign + delta*self.b)/(1.*number_bias_unit) * relevance_adj

        wr = tf.matmul(self.W, relevance_adj)
        br = tf.matmul(bias_term, tf.ones(tf.shape(tf.transpose(self.W))))

        return xin * wr + br

    def rel_z_beta_prop(self, x, relevance, lowest=-1.0, highest=1.0):
        w, v, u = self.W, tf.maximum(0.0, self.W), tf.minimum(0.0, self.W)
        l, h = x * 0 + lowest, x * 0 + highest

        z = tf.matmul(x, w) - (tf.matmul(l, v) + tf.matmul(h, u)) + DIVISION_ADJUSTMENT
        s = relevance / z
        return x * tf.matmul(s, tf.transpose(w)) \
               - l * tf.matmul(s, tf.transpose(v)) - h * tf.matmul(s, tf.transpose(u))

    @staticmethod
    def rel_z_plus_beta_prop(x_p, w_zp, x_b, w_b, relevance, alpha, beta, lowest=-1, highest=1):
        wp_zp = tf.maximum(0.0, w_zp)
        zp_zp = tf.matmul(x_p, wp_zp)

        wn_zp = tf.minimum(0.0, w_zp)
        zn_zp = tf.matmul(x_p, wn_zp)

        w_b, v_b, u_b = w_b, tf.maximum(0.0, w_b), tf.minimum(0.0, w_b)
        l_b, h_b = x_b * 0 + lowest, x_b * 0 + highest
        z_b = tf.matmul(x_b, w_b) - (tf.matmul(l_b, v_b) + tf.matmul(h_b, u_b))

        z = (alpha*zp_zp-beta*zn_zp) + z_b + DIVISION_ADJUSTMENT

        s = relevance / z

        # z-plus
        c_p = tf.matmul(alpha*s, tf.transpose(wp_zp))
        c_n = tf.matmul(-beta*s, tf.transpose(wn_zp))
        r_p = x_p * (c_p + c_n)

        # z-beta
        r_b = x_b * tf.matmul(s, tf.transpose(w_b)) \
              - (l_b * tf.matmul(s, tf.transpose(v_b)) + h_b * tf.matmul(s, tf.transpose(u_b)))

        return r_p, r_b


class ConvolutionalLayer(Layer):
    def __init__(self, input_channels, kernel_size, filters, name, default_weights=None, default_biases=None,
                 padding='SAME'):

        super().__init__(kernel_size + [input_channels, filters], name,
                         default_weights=default_weights, default_biases=default_biases,
                         no_input_activations=np.prod(kernel_size + [input_channels])
                         )

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = [1]*4
        self.padding = padding

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

    def rel_zplus_prop(self, x, relevance, alpha, beta):
        wp = tf.maximum(0.0, self.W)
        wn = tf.minimum(0.0, self.W)

        def compute_c(w, ratio):
            hconv = self.conv_with_w(x, w)

            z = hconv
            s = ratio*relevance / (z + DIVISION_ADJUSTMENT)

            return tf.nn.conv2d_backprop_input(
                tf.shape(x), w,
                out_backprop=s,
                strides=self.strides,
                padding=self.padding
            )

        return x*(compute_c(wp, alpha) + compute_c(wn, -beta))

    def rel_zbeta_prop(self, x, relevance, lowest=-1, highest=1):

        w_neg = tf.minimum(0.0, self.W)
        w_pos = tf.maximum(0.0, self.W)

        l, h = x*0.0+lowest, x*0+highest

        i_act, _ = self.conv(x)
        p_act = self.conv_with_w(l, w_pos)
        n_act = self.conv_with_w(h, w_neg)

        s = relevance / (i_act - (p_act + n_act) + DIVISION_ADJUSTMENT)

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

    def rel_conv_z_plus_beta_prop(self, x_zp, w_zp, x_beta, w_beta, relevance, alpha, beta, lowest=-1, highest=1):
        wp_zp = tf.maximum(0.0, w_zp)
        zp_zp = self.conv_with_w(x_zp, wp_zp)

        wn_zp = tf.minimum(0.0, w_zp)
        zn_zp = self.conv_with_w(x_zp, wn_zp)

        v_b, u_b = tf.maximum(0.0, w_beta), tf.minimum(0.0, w_beta)
        l_b, h_b = x_beta * 0 + lowest, x_beta * 0 + highest

        z_b_x = self.conv_with_w(x_beta, w_beta)
        z_b_lb = self.conv_with_w(l_b, v_b)
        z_b_hb = self.conv_with_w(h_b, u_b)

        z = (alpha*zp_zp-beta*zn_zp) + (z_b_x - (z_b_lb + z_b_hb))

        rel_prop = relevance / (z + DIVISION_ADJUSTMENT)

        def compute_c(shape_h, w, s):
            return tf.nn.conv2d_backprop_input(
                shape_h, w,
                out_backprop=s,
                strides=self.strides,
                padding=self.padding
            )

        # relevance to z_plus
        shape_r_zp = tf.shape(x_zp)
        r_zp = x_zp*(compute_c(shape_r_zp, wp_zp, rel_prop) + compute_c(shape_r_zp, wn_zp, rel_prop))

        # relevance to z-beta
        shape_r_beta = tf.shape(x_beta)
        r_beta = x_beta * compute_c(shape_r_beta, w_beta, rel_prop) \
                 - (l_b * compute_c(shape_r_beta, v_b, rel_prop) + h_b * compute_c(shape_r_beta, u_b, rel_prop))

        return r_zp, r_beta


class PoolingLayer:
    def __init__(self, kernel_size, strides, padding='SAME'):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def pool(self, x):
        return tf.nn.avg_pool(
            x,
            ksize=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1], padding=self.padding
        ) * tf.constant(float(np.size(self.kernel_size)))

    def get_no_variables(self):
        return 0

    def rel_prop(self, x, activations, relevance):
        s = relevance / (activations + DIVISION_ADJUSTMENT)
        c = tf.gradients(activations, x, grad_ys=s)[0]

        return x*c

