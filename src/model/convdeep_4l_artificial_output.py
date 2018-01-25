import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf

from model import base
from model.components.layer import Layer, ConvolutionalLayer, PoolingLayer
from utils import data_provider
from utils import experiment_artifact
from utils import logging as lg
from utils import network_architecture

lg.set_logging()

Architecture = namedtuple('ConvDeep4LArchitecture', ['conv1', 'conv2', 'hidden', 'out1', 'out2', 'recur'])

ARCHITECTURE_NAME = 'convdeep_4l_network'


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: Architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        # define layers
        no_channels = 1

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
        self.ly_input_to_cell = Layer((input_after_conv_layers + architecture.recur, architecture.hidden),
                                      'convdeep_4l__input_to_cell')

        self.ly_output_from_cell = Layer((architecture.hidden, architecture.out1), 'convdeep_4l__output_from_cell')
        self.ly_output_2 = Layer((architecture.out1, architecture.out2), 'convdeep_4l__final_output')

        self.ly_recurrent = Layer((architecture.hidden, architecture.recur), 'convdeep_4l__recurrent')

        self.layers = {
            'conv1': self.ly_conv1,
            'pool1': self.ly_pool1,
            'conv2': self.ly_conv2,
            'pool2': self.ly_pool2,
            'input_to_cell': self.ly_input_to_cell,
            'output_from_cell': self.ly_output_from_cell,
            'output_2': self.ly_output_2,
            'recurrent': self.ly_recurrent
        }

        rr = self.rx

        self.activation_labels = ['conv1','pool1', 'conv2', 'pool2', 'input_to_cell', 'hidden',
                                  'output_from_cell', 'output2', 'recurrent']

        self.activations = namedtuple('Activations', self.activation_labels)\
            (**dict([(k, []) for k in self.activation_labels]))

        self.activations.recurrent.append(self.rx)

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
            xr = tf.concat([in2_reshaped, rr], axis=1)
            self.activations.input_to_cell.append(xr)
            xr_do = tf.nn.dropout(xr, keep_prob=self.keep_prob)

            ha = tf.nn.relu(tf.matmul(xr_do, self.ly_input_to_cell.W) - tf.nn.softplus(self.ly_input_to_cell.b))
            self.activations.hidden.append(ha)

            rr = tf.nn.relu(tf.matmul(ha, self.ly_recurrent.W) - tf.nn.softplus(self.ly_recurrent.b))
            self.activations.recurrent.append(rr)

        last_hidden_activation = self.activations.hidden[-1]
        ha_do = tf.nn.dropout(last_hidden_activation, keep_prob=self.keep_prob)
        last_output_from_cell = tf.nn.relu(tf.matmul(ha_do, self.ly_output_from_cell.W)
                                           - tf.nn.softplus(self.ly_output_from_cell.b))

        self.activations.output_from_cell.append(last_output_from_cell)
        self.y_pred = tf.matmul(last_output_from_cell, self.ly_output_2.W) \
                      - tf.nn.softplus(self.ly_output_2.b)


        # self.setup_loss_and_opt()
        reg_term = tf.constant(0.0)

        for k, v in self.layers.items():
            if hasattr(v, 'W'):
                reg_term = reg_term + tf.reduce_sum(tf.pow(v.W, 2))

        # print(tf.shape(self.y_pred)[0])
        size = (tf.shape(self.y_pred)[0], 1)
        pred_pseudo_class = tf.zeros(size)
        target_pseudo_class = tf.zeros(size)

        y_pred_with_psuedo_class = tf.concat([self.y_pred, pred_pseudo_class], axis=1)
        y_target_with_psuedo_class = tf.concat([self.y_target, target_pseudo_class], axis=1)
        print(y_target_with_psuedo_class.shape)

        self.loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_with_psuedo_class,
                                                    labels=y_target_with_psuedo_class)
            + self.regularizer * reg_term
        )

        optimizer = getattr(tf.train, self.optimizer)
        self.train_op = optimizer(learning_rate=self.lr).minimize(self.loss_op)
        self.init_op = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(self.y_target, 1), tf.argmax(self.y_pred, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', self.loss_op)
        tf.summary.scalar('accuracy', self.accuracy)

        self.summary = tf.summary.merge_all()

        self.y_pred_y_target =  self.y_pred * self.y_target


class Network(base.BaseNetwork):
    def __init__(self, artifact):
        super(Network, self).__init__(artifact)

        self.architecture = Architecture(**network_architecture.parse(artifact.architecture))
        self.dag = Dag(artifact.column_at_a_time, self.data_no_rows, self.data_no_cols, self.architecture,
                       artifact.optimizer, self.architecture.out2)

        self.experiment_artifact = artifact
        self._ = artifact

    def lrp(self, x, y, beta=0.0, alpha=1.0, debug=False):

        with self.get_session() as sess:

            self.dag.setup_variables_for_lrp()

            rel_to_input = [None]*self._.seq_length

            # NOTE: lwr start here
            rel_to_out_from_cell = self.dag.layers['output_2'].rel_z_plus_prop(
                self.dag.activations.output_from_cell[-1],
                self.dag.total_relevance, beta=beta, alpha=alpha
            )
            rel_to_hidden = self.dag.layers['output_from_cell'].rel_z_plus_prop(
                self.dag.activations.hidden[-1],
                rel_to_out_from_cell, beta=beta, alpha=alpha
            )

            rel_to_input_to_cell = self.dag.layers['input_to_cell'].rel_z_plus_prop(
                self.dag.activations.input_to_cell[-1],
                rel_to_hidden, beta=beta, alpha=alpha
            )

            rel_to_recurrent = rel_to_input_to_cell[:, -self.architecture.recur:]

            rel_from_hidden_to_pool2 = tf.reshape(rel_to_input_to_cell[:, :-self.architecture.recur],
                                                  self.dag.shape_pool2_output)

            rel_to_conv2 = self.dag.layers['pool2'].rel_prop(
                self.dag.activations.conv2[-1],
                self.dag.activations.pool2[-1],
                rel_from_hidden_to_pool2
            )

            rel_to_pool1 = self.dag.layers['conv2'].rel_zplus_prop(
                self.dag.activations.pool1[-1],
                rel_to_conv2, beta=beta, alpha=alpha
            )

            rel_to_conv1 = self.dag.layers['pool1'].rel_prop(
                self.dag.activations.conv1[-1],
                self.dag.activations.pool1[-1],
                rel_to_pool1
            )

            rel_to_input[-1] = self.dag.layers['conv1'].rel_zbeta_prop(
                self.dag.x_with_channels[:, :, -self.experiment_artifact.column_at_a_time:, :],
                rel_to_conv1, beta=beta, alpha=alpha
            )

            for i in range(self._.seq_length - 1)[::-1]:

                rel_to_hidden = self.dag.layers['recurrent'].rel_z_plus_prop(
                    self.dag.activations.hidden[i],
                    rel_to_recurrent, beta=beta, alpha=alpha
                )

                rel_to_input_to_cell = self.dag.layers['input_to_cell'].rel_z_plus_prop(
                    self.dag.activations.input_to_cell[i],
                    rel_to_hidden, beta=beta, alpha=alpha
                )

                rel_to_recurrent = rel_to_input_to_cell[:, -self.architecture.recur:]

                rel_from_hidden_to_pool2 = tf.reshape(rel_to_input_to_cell[:, :-self.architecture.recur],
                                                      self.dag.shape_pool2_output
                                                      )

                rel_to_conv2 = self.dag.layers['pool2'].rel_prop(
                    self.dag.activations.conv2[i],
                    self.dag.activations.pool2[i],
                    rel_from_hidden_to_pool2
                )

                rel_to_pool1 = self.dag.layers['conv2'].rel_zplus_prop(
                    self.dag.activations.pool1[i],
                    rel_to_conv2, beta=beta, alpha=alpha
                )

                rel_to_conv1 = self.dag.layers['pool1'].rel_prop(
                    self.dag.activations.conv1[i],
                    self.dag.activations.pool1[i],
                    rel_to_pool1
                )

                c_i = self._.column_at_a_time * i
                c_j = c_i + self._.column_at_a_time

                rel_to_input[i] = self.dag.layers['conv1'].rel_zbeta_prop(
                    self.dag.x_with_channels[:, :, c_i:c_j, :],
                    rel_to_conv1, beta=beta, alpha=alpha
                )

            pred, heatmaps = self._build_heatmap(sess, x, y,
                                                 rr_of_pixels=rel_to_input, debug=debug)
        return pred, heatmaps

