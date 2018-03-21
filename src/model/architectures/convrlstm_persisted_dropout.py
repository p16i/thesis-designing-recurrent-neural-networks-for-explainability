from collections import namedtuple

import tensorflow as tf
import logging

from model.architectures import base, convdeep, convrlstm
from model.components.layer import Layer, ConvolutionalLayer, PoolingLayer
from utils import logging as lg
from utils import network_architecture

import model.components.tfx as tfx
import numpy as np

lg.set_logging()

Architecture = namedtuple('Architecture', ['conv1', 'conv2', 'in1', 'recur', 'out2'])


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        # define conv layers
        dummy_in1 = tf.constant(0.0, shape=[1, dims, no_input_cols, convdeep.NO_CHANNELS])
        self.ly_conv1 = ConvolutionalLayer(name='conv1',
                                           input_channels=convdeep.NO_CHANNELS,
                                           **architecture.conv1['conv'])

        self.ly_pool1 = PoolingLayer(**architecture.conv1['pooling'])

        self.ly_conv2 = ConvolutionalLayer(name='conv2',
                                           input_channels=architecture.conv1['conv']['filters'],
                                           **architecture.conv2['conv'])

        self.ly_pool2 = PoolingLayer(**architecture.conv2['pooling'])

        _, cin1 = self.ly_conv1.conv(dummy_in1)
        self.shape_conv1_output = [-1] + cin1.get_shape().as_list()[1:]  # ignore batch_size
        logging.info('conv1 shape ')
        logging.info(self.shape_conv1_output)

        dummy_in2 = self.ly_pool1.pool(cin1)
        self.shape_pool1_output = [-1] + dummy_in2.get_shape().as_list()[1:]
        logging.info('pool1 shape ')
        logging.info(self.shape_pool1_output)

        _, cin2 = self.ly_conv2.conv(dummy_in2)
        self.shape_conv2_output = [-1] + cin2.get_shape().as_list()[1:]
        logging.info('conv2 shape ')
        logging.info(self.shape_conv2_output)

        dummy_in3 = self.ly_pool2.pool(cin2)
        self.shape_pool2_output = [-1] + dummy_in3.get_shape().as_list()[1:]

        logging.info('Output dims after conv and pooling layers')
        logging.info(dummy_in3.shape)

        input_after_conv_layers = int(np.prod(dummy_in3.shape))

        # define fc layers
        self.ly_input = Layer((input_after_conv_layers, architecture.in1), 'input_1')

        self.ly_input_gate = Layer((architecture.in1 + architecture.recur, architecture.recur), 'input_gate',
                                   default_biases=tf.Variable(tf.zeros([1, architecture.recur])))
        self.ly_forget_gate = Layer((architecture.in1 + architecture.recur, architecture.recur), 'forget_gate',
                                    default_biases=tf.Variable(tf.zeros([1, architecture.recur])))
        self.ly_output_gate = Layer((architecture.in1 + architecture.recur, architecture.recur), 'output_gate',
                                    default_biases=tf.Variable(tf.zeros([1, architecture.recur])))

        self.ly_new_cell_state = Layer((architecture.in1 + architecture.recur, architecture.recur), 'output_gate')
        self.ly_output = Layer((architecture.recur, architecture.out2), 'output')

        self.layers = {
            'conv1': self.ly_conv1,
            'pool1': self.ly_pool1,
            'conv2': self.ly_conv2,
            'pool2': self.ly_pool2,
            'input_1': self.ly_input,
            'input_gate': self.ly_input_gate,
            'forget_gate': self.ly_forget_gate,
            'output_gate': self.ly_output_gate,
            'new_cell_state': self.ly_new_cell_state,
            'output': self.ly_output
        }

        ct = tf.zeros([tf.shape(self.x)[0], architecture.recur])
        ht = tf.zeros([tf.shape(self.x)[0], architecture.recur])

        keys = ['conv1', 'pool1', 'conv2', 'pool2', 'pool2_reshaped', 'prev_ct_from_fg', 'ct_from_ig', 'xr', 'ct', 'ht']
        self.activations = namedtuple('activations', keys)(**dict(map(lambda k: (k, []), keys)))

        ht_do_mark = None
        in1_do_mark = None

        # define  dag
        for i in range(0, max_seq_length, no_input_cols):
            x_4d = self.x_with_channels[:, :, i:i + no_input_cols, :]
            _, in1 = self.ly_conv1.conv(x_4d)
            self.activations.conv1.append(in1)

            pin1 = self.ly_pool1.pool(in1)
            self.activations.pool1.append(pin1)

            _, in2 = self.ly_conv2.conv(pin1)
            self.activations.conv2.append(in2)

            pin2 = self.ly_pool2.pool(in2)
            self.activations.pool2.append(pin2)

            in2_reshaped = tf.reshape(pin2, [-1, input_after_conv_layers])
            self.activations.pool2_reshaped.append(in2_reshaped)

            in1, in1_do_mark = tfx.dropout_with_mark_returned(
                tf.nn.relu(tf.matmul(in2_reshaped, self.ly_input.W) - tf.nn.softplus(self.ly_input.b)),
                keep_prob=self.keep_prob, binary_mark_tensor=in1_do_mark)

            xr = tf.concat([in1, ht], axis=1)
            self.activations.xr.append(xr)

            ig = tf.sigmoid(tf.matmul(xr, self.ly_input_gate.W) + self.ly_input_gate.b)
            fg = tf.sigmoid(tf.matmul(xr, self.ly_forget_gate.W) + self.ly_forget_gate.b)
            og = tf.sigmoid(tf.matmul(xr, self.ly_output_gate.W) + self.ly_output_gate.b)

            new_cell_state = tf.nn.relu(
                tf.matmul(xr, self.ly_new_cell_state.W) - tf.nn.softplus(self.ly_new_cell_state.b))

            prev_ct_from_fg = tf.multiply(fg, ct)
            self.activations.prev_ct_from_fg.append(prev_ct_from_fg)

            ct_from_ig = tf.multiply(ig, new_cell_state)
            self.activations.ct_from_ig.append(ct_from_ig)

            ct = ct_from_ig + prev_ct_from_fg
            self.activations.ct.append(ct)

            ht, ht_do_mark = tfx.dropout_with_mark_returned(tf.multiply(og, ct), keep_prob=self.keep_prob,
                                                            binary_mark_tensor=ht_do_mark)
            self.activations.ht.append(ht)

        self.y_pred = tf.matmul(ht, self.ly_output.W) - tf.nn.softplus(self.ly_output.b)

        self.setup_loss_and_opt()


class Network(convrlstm.Network):
    pass
