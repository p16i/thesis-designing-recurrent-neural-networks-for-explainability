from collections import namedtuple

import tensorflow as tf

from model.architectures import base
from model.components.layer import Layer
from utils import logging as lg
from utils import network_architecture

lg.set_logging()

Architecture = namedtuple('Architecture', ['in1', 'recur', 'out'])


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        # define layers
        self.ly_input = Layer((dims * no_input_cols, architecture.in1), 'input_1')

        self.ly_input_gate = Layer((architecture.in1 + architecture.recur, architecture.recur), 'input_gate',
                                   default_biases=tf.Variable(tf.zeros([1, architecture.recur])))
        self.ly_forget_gate = Layer((architecture.in1 + architecture.recur, architecture.recur), 'forget_gate',
                                    default_biases=tf.Variable(tf.zeros([1, architecture.recur])))
        self.ly_output_gate = Layer((architecture.in1 + architecture.recur, architecture.recur), 'output_gate',
                                    default_biases=tf.Variable(tf.zeros([1, architecture.recur])))

        self.ly_new_cell_state = Layer((architecture.in1 + architecture.recur, architecture.recur), 'output_gate')
        self.ly_output = Layer((architecture.recur, architecture.out), 'output')

        self.layers = {
            'input_1': self.ly_input,
            'input_gate': self.ly_input_gate,
            'forget_gate': self.ly_forget_gate,
            'output_gate': self.ly_output_gate,
            'new_cell_state': self.ly_new_cell_state,
            'output': self.ly_output
        }

        print('no. variables %d' % self.no_variables())

        ct = tf.zeros([tf.shape(self.x)[0], architecture.recur])
        ht = tf.zeros([tf.shape(self.x)[0], architecture.recur])

        # define  dag
        for i in range(0, max_seq_length, no_input_cols):
            ii = tf.reshape(self.x[:, :, i:i + no_input_cols], [-1, no_input_cols * dims])
            in1 = tf.nn.dropout(tf.nn.relu(tf.matmul(ii, self.ly_input.W) - tf.nn.softplus(self.ly_input.b)),
                                keep_prob=self.keep_prob)

            xr = tf.concat([in1, ht], axis=1)

            ig = tf.nn.dropout(tf.sigmoid(tf.matmul(xr, self.ly_input_gate.W) + self.ly_input_gate.b),
                               keep_prob=self.keep_prob)
            fg = tf.nn.dropout(tf.sigmoid(tf.matmul(xr, self.ly_forget_gate.W) + self.ly_forget_gate.b),
                               keep_prob=self.keep_prob)
            og = tf.nn.dropout(tf.sigmoid(tf.matmul(xr, self.ly_output_gate.W) + self.ly_output_gate.b),
                               keep_prob=self.keep_prob)

            new_cell_state = tf.nn.dropout(tf.nn.relu(
                tf.matmul(xr, self.ly_new_cell_state.W) - tf.nn.softplus(self.ly_new_cell_state.b)),
                keep_prob=self.keep_prob)

            prev_ct_from_fg = tf.multiply(fg, ct)
            ct_from_ig = tf.multiply(ig, new_cell_state)

            ct = ct_from_ig + prev_ct_from_fg

            ht = tf.nn.dropout(tf.multiply(og, ct), keep_prob=self.keep_prob)

        self.y_pred = tf.matmul(ht, self.ly_output.W) - tf.nn.softplus(self.ly_output.b)

        self.setup_loss_and_opt()


class Network(base.BaseNetwork):
    def __init__(self, artifact):
        super(Network, self).__init__(artifact)

        self.architecture = Architecture(**network_architecture.parse(artifact.architecture))
        self.dag = Dag(artifact.column_at_a_time, self.data_no_rows, self.data_no_cols,
                       self.architecture, artifact.optimizer, self.architecture.out2)

        self.experiment_artifact = artifact
        self._ = artifact

    def lrp(self, x, y, beta=0.0, alpha=1.0, debug=False):
        x_3d = x.reshape(-1, self.data_no_rows, self.data_no_cols)
        with self.get_session() as sess:
            # lwr start here
            self.dag.setup_variables_for_lrp()
            rel_to_input = [None] * self._.seq_length

            rel_to_output_from_cell = self.dag.layers['output_2'].rel_z_plus_prop(
                self.dag.output_from_cell_activations[-1],
                self.dag.total_relevance, beta=beta, alpha=alpha
            )

            rel_to_hidden = self.dag.layers['output_from_cell'].rel_z_plus_prop(
                self.dag.ha_activations[-1],
                rel_to_output_from_cell, beta=beta, alpha=alpha
            )

            rel_to_input_to_cell = self.dag.layers['input_to_cell'].rel_z_plus_prop(
                self.dag.input_to_cell_activations[-1],
                rel_to_hidden, beta=beta, alpha=alpha
            )

            rel_to_recurrent = rel_to_input_to_cell[:, -self.architecture.recur:]

            rel_from_hidden_to_in1 = rel_to_input_to_cell[:, :-self.architecture.recur]

            rel_to_input[-1] = self.dag.layers['input_1'].rel_z_beta_prop(
                tf.reshape(self.dag.x[:, :, -self.experiment_artifact.column_at_a_time:], shape=[x_3d.shape[0], -1]),
                rel_from_hidden_to_in1
            )

            for i in range(self._.seq_length - 1)[::-1]:
                rel_to_hidden = self.dag.layers['recurrent'].rel_z_plus_prop(
                    self.dag.ha_activations[i],
                    rel_to_recurrent, beta=beta, alpha=alpha
                )

                rel_to_input_to_cell = self.dag.layers['input_to_cell'].rel_z_plus_prop(
                    self.dag.input_to_cell_activations[i],
                    rel_to_hidden, beta=beta, alpha=alpha
                )

                rel_to_recurrent = rel_to_input_to_cell[:, -self.architecture.recur:]

                rel_from_hidden_to_in1 = rel_to_input_to_cell[:, :-self.architecture.recur]

                c_i = self._.column_at_a_time * i
                c_j = c_i + self._.column_at_a_time

                rel_to_input[i] = self.dag.layers['input_1'].rel_z_beta_prop(
                    tf.reshape(self.dag.x[:, :, c_i:c_j], shape=[x_3d.shape[0], -1]),
                    rel_from_hidden_to_in1
                )

            pred, heatmaps = self._build_heatmap(sess, x, y, rr_of_pixels=rel_to_input, debug=debug)

        return pred, heatmaps
