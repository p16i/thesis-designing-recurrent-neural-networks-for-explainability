from collections import namedtuple

import numpy as np
import tensorflow as tf

from model.architectures import base
from model.components.layer import Layer
from utils import experiment_artifact
from utils import logging as lg
from utils import network_architecture

lg.set_logging()


Architecture = namedtuple('Architecture', ['size', 'out', 'recur'])


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: Architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        with tf.variable_scope('LSTM') as vs:
            ly_input_gate = Layer((no_input_cols * dims + architecture.size, architecture.size),
                                  default_biases=tf.Variable(tf.zeros([1, architecture.size])),
                                  name='input_gate')
            ly_forget_gate = Layer((no_input_cols * dims + architecture.size, architecture.size),
                                   default_biases=tf.Variable(tf.zeros([1, architecture.size])),
                                   name='forget_gate')
            ly_output_gate = Layer((no_input_cols * dims + architecture.size, architecture.size),
                                   default_biases=tf.Variable(tf.zeros([1, architecture.size])),
                                   name='output_gate')

            ly_new_cell_state = Layer((no_input_cols * dims + architecture.size, architecture.size),
                                      default_biases=tf.Variable(tf.zeros([1, architecture.size])),
                                      name='new_cell_state')

            ly_final_output = Layer((architecture.size, architecture.out),
                                    default_biases=tf.Variable(tf.zeros(([1, architecture.out]))),
                                    name='lstm_final_output')

            self.layers = {
                'input_gate': ly_input_gate,
                'forget_gate': ly_forget_gate,
                'output_gate': ly_output_gate,
                'final_output': ly_final_output,
                'new_cell_state': ly_new_cell_state
            }

            print('No. of variables %d' % self.no_variables())

            self.activation_labels = ['input_gate', 'forget_gate', 'output_gate', 'xh',
                                      'new_cell_state', 'output']

            self.activations = namedtuple('Activations', self.activation_labels) \
                (**dict([(k, []) for k in self.activation_labels]))

            ct = tf.zeros([tf.shape(self.x)[0], architecture.size])
            ht = tf.zeros([tf.shape(self.x)[0], architecture.size])

            # define  dag
            for i in range(0, max_seq_length, no_input_cols):
                xt = tf.reshape(self.x[:, :, i:i + no_input_cols], [-1, no_input_cols * dims])

                xh = tf.concat([xt, ht], axis=1)
                self.activations.xh.append(xh)

                ig = tf.sigmoid(tf.matmul(xh, ly_input_gate.W) + ly_input_gate.b)
                ig_do = tf.nn.dropout(ig, keep_prob=self.keep_prob)
                self.activations.input_gate.append(ig)

                fg = tf.sigmoid(tf.matmul(xh, ly_forget_gate.W) + ly_forget_gate.b)
                fg_do = tf.nn.dropout(fg, keep_prob=self.keep_prob)
                self.activations.forget_gate.append(fg)

                og = tf.sigmoid(tf.matmul(xh, ly_output_gate.W) + ly_output_gate.b)
                og_do = tf.nn.dropout(og, keep_prob=self.keep_prob)
                self.activations.output_gate.append(og)

                new_c = tf.tanh(tf.matmul(xh, ly_new_cell_state.W) + ly_new_cell_state.b)
                new_c_do = tf.nn.dropout(new_c, keep_prob=self.keep_prob)

                ct = ct*fg_do + new_c_do*ig_do
                ct_do = tf.nn.dropout(ct, keep_prob=self.keep_prob)
                self.activations.new_cell_state.append(ct)

                ht = og_do*tf.tanh(ct_do)
                self.activations.output.append(ht)

                # self.activations.input_to_cell.append(xr)
                # xr_do = tf.nn.dropout(xr, keep_prob=self.keep_prob)
                # output, state = lstm(ii, state)

            ht_do = tf.nn.dropout(ht, keep_prob=self.keep_prob)
            self.y_pred = tf.matmul(ht_do, ly_final_output.W) + ly_final_output.b

            self.setup_loss_and_opt()


class Network(base.BaseNetwork):
    def __init__(self, artifact: experiment_artifact.Artifact):
        super(Network, self).__init__(artifact)

        self.architecture = Architecture(**network_architecture.parse(artifact.architecture))

        tf.reset_default_graph()

        self.dag = Dag(artifact.column_at_a_time,
                       self.data_no_rows, self.data_no_cols,
                       self.architecture, artifact.optimizer, self.architecture.out)

        self.experiment_artifact = artifact
        self._ = artifact

        self.name = 'lstm'

    def rel_guided_backprop(self, x, y, debug=False):
        return np.zeros(y.shape[0]), np.zeros(x.shape)

    def rel_lrp_deep_taylor(self, x, y, debug=False):
        return self.rel_guided_backprop(x, y, debug)

    def lrp(self, x, y, alpha=1.0, beta=0.0, debug=False):

        with self.get_session() as sess:

            self.dag.setup_variables_for_lrp()

            rel_to_input = [None]*self._.seq_length

            # NOTE: lwr start here
            rel_from_output_to_cell = self.dag.layers['final_output'].rel_z_plus_prop(
                self.dag.activations.output[-1],
                self.dag.total_relevance, beta=beta, alpha=alpha
            )

            proportion_to_ct = self.dag.activations.forget_gate[-1] / \
                              (self.dag.activations.input_gate[-1] + self.dag.activations.forget_gate[-1])

            rel_to_ct = proportion_to_ct*rel_from_output_to_cell

            rel_to_xh = self.dag.layers['new_cell_state'].rel_z_plus_prop(
                self.dag.activations.xh[-1],
                (1-proportion_to_ct)*rel_from_output_to_cell, alpha=alpha, beta=beta
            )

            rel_to_input[-1] = rel_to_xh[:, :-self.architecture.size]
            rel_to_h = rel_to_xh[:, -self.architecture.size:]

            rel_to_ct = rel_to_ct + rel_to_h


            for i in range(self._.seq_length - 1)[::-1]:
                proportion_to_ct = self.dag.activations.forget_gate[i] / \
                                   (self.dag.activations.input_gate[i] + self.dag.activations.forget_gate[i])

                rel_to_ct = proportion_to_ct*rel_to_ct

                rel_to_xh = self.dag.layers['new_cell_state'].rel_z_plus_prop(
                    self.dag.activations.xh[i],
                    (1-proportion_to_ct)*rel_from_output_to_cell, alpha=alpha, beta=beta
                )

                rel_to_input[i] = rel_to_xh[:, :-self.architecture.size]
                rel_to_h = rel_to_xh[:, -self.architecture.size:]

                rel_to_ct = rel_to_ct + rel_to_h

            pred, heatmaps = self._build_heatmap(sess, x, y,
                                                 rr_of_pixels=rel_to_input, debug=debug)
        return pred, heatmaps

