import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf
import copy

from model import base
from model.components import lrp
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
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: Architecture, optimizer):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer)

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

        dummy_in2 = self.ly_pool1.pool(cin1)
        self.shape_pool1_output = [-1] + dummy_in2.get_shape().as_list()[1:]

        _, cin2 = self.ly_conv2.conv(dummy_in2)
        self.shape_conv2_output = [-1] + cin2.get_shape().as_list()[1:]

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

            ha_do = tf.nn.dropout(ha, keep_prob=self.keep_prob)
            ho = tf.nn.relu(tf.matmul(ha_do, self.ly_output_from_cell.W) - tf.nn.softplus(self.ly_output_from_cell.b))
            self.activations.output_from_cell.append(ho)

            ho_do = tf.nn.dropout(ho, keep_prob=self.keep_prob)
            ot = tf.matmul(ho_do, self.ly_output_2.W) - tf.nn.softplus(self.ly_output_2.b)

        self.y_pred = ot

        self.setup_loss_and_opt()


class Network(base.BaseNetwork):
    def __init__(self, artifact):
        super(Network, self).__init__(artifact)

        self.architecture = Architecture(**network_architecture.parse(artifact.architecture))
        self.dag = Dag(artifact.column_at_a_time, 28, 28, self.architecture, artifact.optimizer)

        self.experiment_artifact = artifact
        self._ = artifact

    @staticmethod
    def train(seq_length=1, epoch=1, lr=0.01, batch=100,
              architecture_str='in1:_|in2:_|hidden:_|out1:_|out2:_|--recur:_',
              keep_prob=0.5, verbose=False, output_dir='./experiment-result', optimizer='AdamOptimizer',
              dataset='mnist', regularizer=0.0,
              ):

        experiment_name = experiment_artifact.get_experiment_name('%s-%s-seq-%d--' %
                                                                  (ARCHITECTURE_NAME, dataset, seq_length))
        logging.debug('Train s9 4 layers network')
        logging.debug('Experiment name : %s' % experiment_name)

        data = data_provider.get_data(dataset)

        # no.rows and cols
        dims, max_seq_length = data.train2d.x.shape[1:]
        architecture = Architecture(**network_architecture.parse(architecture_str))
        logging.debug('Network architecture')
        logging.debug(architecture)

        logging.debug('Optimizer %s' % optimizer)

        no_input_cols = max_seq_length // seq_length
        logging.debug('Training %d columns at a time' % no_input_cols)

        dag = Dag(no_input_cols, dims, max_seq_length, architecture, optimizer)

        with tf.Session() as sess:
            sess.run(dag.init_op)
            step = 1
            for i in range(epoch):
                logging.debug('epoch %d' % (i + 1))
                for bx, by in data.train2d.get_batch(no_batch=batch):

                    rx0 = np.zeros((batch, architecture.recur))
                    sess.run(dag.train_op,
                             feed_dict={dag.x: bx, dag.y_target: by, dag.rx: rx0, dag.lr: lr,
                                        dag.keep_prob: keep_prob, dag.regularizer: regularizer})

                    if (step % 1000 == 0 or step < 10) and verbose:
                        rx0 = np.zeros((len(by), architecture.recur))
                        acc, loss = sess.run([dag.accuracy, dag.loss_op],
                                             feed_dict={dag.x: bx, dag.y_target: by, dag.rx: rx0, dag.keep_prob: 1,
                                                        dag.regularizer: regularizer
                                                        })

                        rx0 = np.zeros((len(data.val2d.y), architecture.recur))
                        acc_val = sess.run(dag.accuracy, feed_dict={dag.x: data.val2d.x, dag.y_target: data.val2d.y,
                                                                    dag.rx: rx0, dag.keep_prob: 1})
                        logging.debug('step %d : current train batch acc %f, loss %f | val acc %f'
                                     % (step, acc, loss, acc_val))

                    step = step + 1

            logging.debug('Calculating test accuracy')
            rx0 = np.zeros((len(data.test2d.y), architecture.recur))
            acc = float(sess.run(dag.accuracy,
                                 feed_dict={dag.x: data.test2d.x, dag.y_target: data.test2d.y,
                                            dag.rx: rx0, dag.keep_prob: 1}))

            rx0 = np.zeros((len(data.val2d.y), architecture.recur))
            val_acc = float(sess.run(dag.accuracy, feed_dict={dag.x: data.val2d.x, dag.y_target: data.val2d.y,
                                                        dag.rx: rx0, dag.keep_prob: 1}))

            res = dict(
                experiment_name=experiment_name,
                seq_length=seq_length,
                epoch=epoch,
                column_at_a_time=no_input_cols,
                batch=batch,
                accuracy=acc,
                lr=lr,
                architecture=architecture_str,
                architecture_name=ARCHITECTURE_NAME,
                dims=dims,
                max_seq_length=max_seq_length,
                keep_prob=keep_prob,
                optimizer=optimizer,
                val_accuracy=val_acc,
                dataset=dataset,
                regularizer=regularizer
            )

            logging.debug('\n%s\n', lg.tabularize_params(res))

            output_dir = '%s/%s' % (output_dir, experiment_name)

            return experiment_artifact.save_artifact(sess, res, output_dir=output_dir)

    def lrp(self, x, factor=1, debug=False):

        x_3d = x.reshape(-1, 28, 28)
        with self.get_session() as sess:

            self.dag.setup_variables_for_lrp()

            rel_to_input = [None]*self._.seq_length

            # NOTE: lwr start here
            rel_to_out_from_cell = self.dag.layers['output_2'].rel_z_plus_prop(
                self.dag.activations.output_from_cell[-1],
                self.dag.total_relevance, factor=factor
            )
            rel_to_hidden = self.dag.layers['output_from_cell'].rel_z_plus_prop(
                self.dag.activations.hidden[-1],
                rel_to_out_from_cell, factor=factor
            )

            rel_to_input_to_cell = self.dag.layers['input_to_cell'].rel_z_plus_prop(
                self.dag.activations.input_to_cell[-1],
                rel_to_hidden, factor=factor
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
                rel_to_conv2, factor=factor
            )

            rel_to_conv1 = self.dag.layers['pool1'].rel_prop(
                self.dag.activations.conv1[-1],
                self.dag.activations.pool1[-1],
                rel_to_pool1
            )

            rel_to_input[-1] = self.dag.layers['conv1'].rel_zbeta_prop(
                self.dag.x_with_channels[:, :, -self.experiment_artifact.column_at_a_time:, :],
                rel_to_conv1
            )

            for i in range(self._.seq_length - 1)[::-1]:

                rel_to_hidden = self.dag.layers['recurrent'].rel_z_plus_prop(
                    self.dag.activations.hidden[i],
                    rel_to_recurrent, factor=factor
                )

                rel_to_input_to_cell = self.dag.layers['input_to_cell'].rel_z_plus_prop(
                    self.dag.activations.input_to_cell[i],
                    rel_to_hidden, factor=factor
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
                    rel_to_conv2, factor=factor
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
                    rel_to_conv1, factor=factor
                )

            pred, heatmaps = self._build_heatmap(sess, x,
                                                 rr_of_pixels=rel_to_input, debug=debug)
        return pred, heatmaps

