import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf

from model.architectures import base, convdeep
from model.components.layer import Layer, ConvolutionalLayer, PoolingLayer
from utils import logging as lg
from utils import network_architecture

import model.components.tfx as tfx

lg.set_logging()

Architecture = namedtuple('ConvDeep4LArchitecture', ['conv1', 'conv2', 'in1', 'hidden', 'out1', 'out2', 'recur'])

ARCHITECTURE_NAME = 'convdeep_4l_network'


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: Architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        # define layers
        no_channels = convdeep.NO_CHANNELS

        dummy_in1 = tf.constant(0.0, shape=[1, dims, no_input_cols, no_channels])
        self.ly_conv1 = ConvolutionalLayer(name='convdeep_4l__conv1',
                                           input_channels=no_channels,
                                           **architecture.conv1['conv'])

        self.ly_pool1 =PoolingLayer(**architecture.conv1['pooling'])

        self.ly_conv2 = ConvolutionalLayer(name='convdeep_4l__conv2',
                                           input_channels=architecture.conv1['conv']['filters'],
                                           **architecture.conv2['conv'])

        self.ly_pool2 =PoolingLayer(**architecture.conv2['pooling'])

        _, cin1 = self.ly_conv1.conv(dummy_in1)
        self.shape_conv1_output = [-1] + cin1.get_shape().as_list()[1:] #ignore batch_size
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

        self.ly_input1 = Layer((input_after_conv_layers, architecture.in1), 'convdeep_4l__in1')

        self.ly_input_to_cell = Layer((architecture.in1 + architecture.recur, architecture.hidden),
                                      'convdeep_4l__input_to_cell')

        self.ly_output_from_cell = Layer((architecture.hidden, architecture.out1), 'convdeep_4l__output_from_cell')
        self.ly_output_2 = Layer((architecture.out1, architecture.out2), 'convdeep_4l__final_output')

        self.ly_recurrent = Layer((architecture.hidden, architecture.recur), 'convdeep_4l__recurrent')

        self.layers = {
            'conv1': self.ly_conv1,
            'pool1': self.ly_pool1,
            'conv2': self.ly_conv2,
            'pool2': self.ly_pool2,
            'input_1': self.ly_input1,
            'input_to_cell': self.ly_input_to_cell,
            'output_from_cell': self.ly_output_from_cell,
            'output_2': self.ly_output_2,
            'recurrent': self.ly_recurrent
        }

        np.random.seed(71)
        mark = (np.random.uniform(0, 1, (architecture.hidden, architecture.recur)) < 0.1)*np.power(10, 0.5)
        mark2 = (np.random.uniform(0, 1, (architecture.recur, architecture.hidden)) < 0.1)*np.power(10, 0.5)
        rr_to_hidden_mark = np.ones((architecture.in1 + architecture.recur, architecture.hidden))
        rr_to_hidden_mark[-architecture.recur:, :] = mark2

        rr = self.rx

        self.activation_labels = ['conv1','pool1', 'conv2', 'pool2', 'pool2_reshaped', 'input_to_cell', 'hidden',
                                  'output_from_cell', 'output2', 'recurrent']

        self.activations = namedtuple('Activations', self.activation_labels)\
            (**dict([(k, []) for k in self.activation_labels]))

        self.activations.recurrent.append(self.rx)

        ha_do_mark = None

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

            xin = tf.nn.relu(tf.matmul(in2_reshaped, self.ly_input1.W) - tf.nn.softplus(self.ly_input1.b))
            xin_do = tf.nn.dropout(xin, keep_prob=self.keep_prob)

            xr = tf.concat([xin_do, rr], axis=1)
            self.activations.input_to_cell.append(xr)
            xr_do = tf.nn.dropout(xr, keep_prob=self.keep_prob)

            ha = tf.nn.relu(tf.matmul(xr_do, self.ly_input_to_cell.W*rr_to_hidden_mark) -
                            tf.nn.softplus(self.ly_input_to_cell.b))
            self.activations.hidden.append(ha)

            ha_do, ha_do_mark = tfx.dropout_with_mark_returned(ha, keep_prob=self.keep_prob,
                                                               binary_mark_tensor=ha_do_mark)
            rr = tf.nn.relu(tf.matmul(ha_do, self.ly_recurrent.W*mark) - tf.nn.softplus(self.ly_recurrent.b))
            self.activations.recurrent.append(rr)

        last_hidden_activation = self.activations.hidden[-1]
        ha_do = tf.nn.dropout(last_hidden_activation, keep_prob=self.keep_prob)
        last_output_from_cell = tf.nn.relu(tf.matmul(ha_do, self.ly_output_from_cell.W)
                                           - tf.nn.softplus(self.ly_output_from_cell.b))

        self.activations.output_from_cell.append(last_output_from_cell)
        self.y_pred = tf.matmul(last_output_from_cell, self.ly_output_2.W) \
                      - tf.nn.softplus(self.ly_output_2.b)

        self.setup_loss_and_opt()


class Network(convdeep.Network):
    pass
