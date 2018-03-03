import logging
from collections import namedtuple

import tensorflow as tf

from model.architectures import base
from model.components.layer import Layer, ConvolutionalLayer, PoolingLayer
from utils import logging as lg
from utils import network_architecture

lg.set_logging()

Architecture = namedtuple('ConvDeep4LArchitecture', ['conv1', 'conv2', 'hidden', 'out1', 'out2', 'recur'])

ARCHITECTURE_NAME = 'tutorial_network_network'


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: Architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        # define layers
        no_channels = 1

        self.ly_conv1 = ConvolutionalLayer(name='tutorial_network__conv1',
                                           input_channels=no_channels, kernel_size=[5, 5], filters=10
                                           )

        self.ly_pool1 = PoolingLayer(kernel_size=[2, 2], strides=[2, 2])

        self.ly_conv2 = ConvolutionalLayer(name='tutorial_network__conv2', padding='VALID',
                                           input_channels=10, kernel_size=[5, 5], filters=25)

        self.ly_pool2 =PoolingLayer(kernel_size=[2, 2], strides=[2, 2])

        self.ly_conv3 = ConvolutionalLayer(name='tutorial_network__conv3', padding='VALID',
                                           input_channels=25, kernel_size=[4, 4], filters=100)

        self.ly_pool3 = PoolingLayer(kernel_size=[2, 2], strides=[2, 2])


        #cp1
        _, cin1 = self.ly_conv1.conv(self.x_with_channels)
        self.shape_conv1_output = [-1] + cin1.get_shape().as_list()[1:] #ignore batch_size
        logging.info('conv1 shape ')
        logging.info(self.shape_conv1_output)

        dummy_in1 = self.ly_pool1.pool(cin1)
        self.shape_pool1_output = [-1] + dummy_in1.get_shape().as_list()[1:]
        logging.info('pool1 shape ')
        logging.info(self.shape_pool1_output)

        #cp2
        _, cin2 = self.ly_conv2.conv(dummy_in1)
        self.shape_conv2_output = [-1] + cin2.get_shape().as_list()[1:]
        logging.info('conv2 shape ')
        logging.info(self.shape_conv2_output)

        dummy_in2 = self.ly_pool2.pool(cin2)
        logging.info(dummy_in2.get_shape())

        # cp3
        _, cin3 = self.ly_conv3.conv(dummy_in2)
        logging.info('conv3 shape')
        logging.info(cin3.get_shape())

        dummy_in3 = self.ly_pool3.pool(cin3)
        logging.info('pool3 shape')
        logging.info(dummy_in3.get_shape())

        self.layers = dict()


        dummy_in3_squeezed = tf.reshape(dummy_in3, shape=[-1, 100])

        self.ly_last = Layer((100, 10), 'tutorial_network__last_layer')
        self.activations = {
            'pool3_squeezed': dummy_in3_squeezed,
            'pool3': dummy_in3,
            'conv3': cin3,
            'pool2': dummy_in2,
            'conv2': cin2,
            'pool1': dummy_in1,
            'conv1': cin1
        }

        logging.info(dummy_in3_squeezed.get_shape())

        # self.y_pred = tf.matmul(self.t)
        self.y_pred = tf.matmul(dummy_in3_squeezed, self.ly_last.W) - tf.nn.softplus(self.ly_last.b)
        # logging.info('y_pred size')
        # logging.info(self.y_pred.get_shape())

        # self.shape_pool2_output = [-1] + dummy_in3.get_shape().as_list()[1:]
        #
        # logging.info('Output dims after conv and pooling layers')
        # logging.info(dummy_in3.shape)
        #
        # input_after_conv_layers = int(np.prod(dummy_in3.shape))
        # self.ly_input_to_cell = Layer((input_after_conv_layers + architecture.recur, architecture.hidden),
        #                               'tutorial_network__input_to_cell')
        #
        # self.ly_output_from_cell = Layer((architecture.hidden, architecture.out1), 'tutorial_network__output_from_cell')
        # self.ly_output_2 = Layer((architecture.out1, architecture.out2), 'tutorial_network__final_output')
        #
        # self.ly_recurrent = Layer((architecture.hidden, architecture.recur), 'tutorial_network__recurrent')
        #
        # self.layers = {
        #     'conv1': self.ly_conv1,
        #     'pool1': self.ly_pool1,
        #     'conv2': self.ly_conv2,
        #     'pool2': self.ly_pool2,
        #     'input_to_cell': self.ly_input_to_cell,
        #     'output_from_cell': self.ly_output_from_cell,
        #     'output_2': self.ly_output_2,
        #     'recurrent': self.ly_recurrent
        # }
        #
        # rr = self.rx
        #
        # self.activation_labels = ['conv1','pool1', 'conv2', 'pool2', 'input_to_cell', 'hidden',
        #                           'output_from_cell', 'output2', 'recurrent']
        #
        # self.activations = namedtuple('Activations', self.activation_labels)\
        #     (**dict([(k, []) for k in self.activation_labels]))
        #
        # self.activations.recurrent.append(self.rx)
        #
        # # define  dag
        # for i in range(0, max_seq_length, no_input_cols):
        #     x_4d = self.x_with_channels[:, :, i:i + no_input_cols, :]
        #     _, in1 = self.ly_conv1.conv(x_4d)
        #     self.activations.conv1.append(in1)
        #
        #     pin1 = self.ly_pool1.pool(in1)
        #     self.activations.pool1.append(pin1)
        #
        #     _, in2 = self.ly_conv2.conv(pin1)
        #     self.activations.conv2.append(in2)
        #
        #     pin2 = self.ly_pool2.pool(in2)
        #     self.activations.pool2.append(pin2)
        #
        #     in2_reshaped = tf.reshape(pin2, [-1, input_after_conv_layers])
        #     xr = tf.concat([in2_reshaped, rr], axis=1)
        #     self.activations.input_to_cell.append(xr)
        #     xr_do = tf.nn.dropout(xr, keep_prob=self.keep_prob)
        #
        #     ha = tf.nn.relu(tf.matmul(xr_do, self.ly_input_to_cell.W) - tf.nn.softplus(self.ly_input_to_cell.b))
        #     self.activations.hidden.append(ha)
        #
        #     rr = tf.nn.relu(tf.matmul(ha, self.ly_recurrent.W) - tf.nn.softplus(self.ly_recurrent.b))
        #     self.activations.recurrent.append(rr)
        #
        # last_hidden_activation = self.activations.hidden[-1]
        # ha_do = tf.nn.dropout(last_hidden_activation, keep_prob=self.keep_prob)
        # last_output_from_cell = tf.nn.relu(tf.matmul(ha_do, self.ly_output_from_cell.W)
        #                                    - tf.nn.softplus(self.ly_output_from_cell.b))
        #
        # self.activations.output_from_cell.append(last_output_from_cell)
        # self.y_pred = tf.matmul(last_output_from_cell, self.ly_output_2.W) \
        #               - tf.nn.softplus(self.ly_output_2.b)


        self.setup_loss_and_opt()


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

            rel_last_to_pool3 = self.dag.ly_last.rel_z_plus_prop(
                self.dag.activations['pool3_squeezed'], self.dag.total_relevance,
                alpha=alpha, beta=beta
            )

            rel_pool3_to_conv3 = self.dag.ly_pool3.rel_prop(
                self.dag.activations['conv3'],
                self.dag.activations['pool3'],
                tf.reshape(rel_last_to_pool3, [-1, 1, 1, 100])
            )

            rel_conv3_to_pool2 = self.dag.ly_conv3.rel_zplus_prop(
                self.dag.activations['pool2'], rel_pool3_to_conv3,
                alpha=alpha, beta=beta
            )

            rel_pool2_to_conv2 = self.dag.ly_pool2.rel_prop(
                self.dag.activations['conv2'],
                self.dag.activations['pool2'],
                rel_conv3_to_pool2
            )

            rel_conv2_to_pool1 = self.dag.ly_conv2.rel_zplus_prop(
                self.dag.activations['pool1'], rel_pool2_to_conv2,
                alpha=alpha, beta=beta
            )

            rel_pool1_to_conv1 = self.dag.ly_pool1.rel_prop(
                self.dag.activations['conv1'],
                self.dag.activations['pool1'],
                rel_conv2_to_pool1
            )

            rel_to_input = self.dag.ly_conv1.rel_zbeta_prop(
                self.dag.x_with_channels,
                rel_pool1_to_conv1
            )

            pred, heatmaps = self._build_heatmap(sess, x, y,
                                                 rr_of_pixels=[rel_to_input], debug=debug)
        return pred, heatmaps

