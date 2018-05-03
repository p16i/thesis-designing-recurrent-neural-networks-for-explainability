from collections import namedtuple

import tensorflow as tf

from model.architectures import base
from model.components.layer import Layer
from utils import logging as lg
from utils import network_architecture

import numpy as np

lg.set_logging()

Architecture = namedtuple('Architecture', ['in1', 'recur', 'out2'])


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        # define layers
        self.ly_input = Layer((dims * no_input_cols, architecture.in1), 'input_1')

        self.ly_input_gate = Layer((architecture.in1 + architecture.recur, architecture.recur), 'input_gate',
                                   default_biases=tf.Variable(tf.zeros([1, architecture.recur])))
        self.ly_recurrent_gate = Layer((architecture.in1 + architecture.recur, architecture.recur), 'forget_gate',
                                    default_biases=tf.Variable(tf.zeros([1, architecture.recur])))

        self.ly_new_ht = Layer((architecture.in1 + architecture.recur, architecture.recur), 'ht_candidate')
        self.ly_output = Layer((architecture.recur, architecture.out2), 'output')

        self.layers = {
            'input_1': self.ly_input,
            'input_gate': self.ly_input_gate,
            'recurrent_gate': self.ly_recurrent_gate,
            'new_ht': self.ly_new_ht,
            'output': self.ly_output
        }

        ht = tf.zeros([tf.shape(self.x)[0], architecture.recur])

        keys = ['ht', 'xh', 'xh_scaled', 'ht_candidate_from_prev', 'ht_candidate_from_xh']
        self.activations = namedtuple('activations', keys)(**dict(map(lambda k: (k, []), keys)))

        # define  dag
        for i in range(0, max_seq_length, no_input_cols):
            ii = tf.reshape(self.x[:, :, i:i + no_input_cols], [-1, no_input_cols * dims])
            in1 = tf.nn.dropout(tf.nn.relu(tf.matmul(ii, self.ly_input.W) - tf.nn.softplus(self.ly_input.b)),
                                keep_prob=self.keep_prob)

            xh = tf.concat([in1, ht], axis=1)
            self.activations.xh.append(xh)

            ig = tf.nn.dropout(tf.sigmoid(tf.matmul(xh, self.ly_input_gate.W) + self.ly_input_gate.b),
                               keep_prob=self.keep_prob)
            rg = tf.nn.dropout(tf.sigmoid(tf.matmul(xh, self.ly_recurrent_gate.W) + self.ly_recurrent_gate.b),
                               keep_prob=self.keep_prob)

            xh_scaled = tf.concat([in1, tf.multiply(rg, ht)], axis=1)
            self.activations.xh_scaled.append(xh_scaled)

            ht_candidate = tf.nn.dropout(tf.nn.relu(
                tf.matmul(xh_scaled, self.ly_new_ht.W) - tf.nn.softplus(self.ly_new_ht.b)),
                keep_prob=self.keep_prob)

            ht_candidate_from_xh = tf.multiply(ig, ht_candidate)
            self.activations.ht_candidate_from_xh.append(ht_candidate_from_xh)

            ht_candidate_from_prev = tf.multiply(1-ig, ht)
            self.activations.ht_candidate_from_prev.append(ht_candidate_from_prev)

            ht = tf.nn.dropout(ht_candidate_from_prev + ht_candidate_from_xh, keep_prob=self.keep_prob)
            self.activations.ht.append(ht)

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
            rr_ct = [None]*self._.seq_length

            dag = self.dag

            rel_from_output_to_ht = dag.layers['output'].rel_z_plus_prop(
                dag.activations.ht[-1],
                dag.total_relevance,
                alpha=alpha, beta=beta
            )

            proportion_to_prev_ht = dag.activations.ht_candidate_from_prev[-1] / (dag.activations.ht[-1] + 1e-10)

            rel_from_ht_to_prev_ht = proportion_to_prev_ht * rel_from_output_to_ht
            rel_to_data_ht = (1-proportion_to_prev_ht)*rel_from_output_to_ht

            rel_to_xh = dag.layers['new_ht'].rel_z_plus_prop(
                dag.activations.xh[-1],
                rel_to_data_ht,
                alpha=alpha, beta=beta
            )

            rel_from_xh_to_ht = rel_to_xh[:, -self.architecture.recur:]
            rel_to_in1 = rel_to_xh[:, :-self.architecture.recur]

            rel_to_input[-1] = self.dag.layers['input_1'].rel_z_beta_prop(
                tf.reshape(self.dag.x[:, :, -self.experiment_artifact.column_at_a_time:],
                           shape=[-1, self._.column_at_a_time * self._.dims]),
                rel_to_in1
            )

            rel_to_prev_ht = rel_from_ht_to_prev_ht + rel_from_xh_to_ht

            for i in range(self._.seq_length - 1)[::-1]:
                proportion_to_prev_ht = dag.activations.ht_candidate_from_prev[i] / (dag.activations.ht[i] + 1e-10)

                rel_from_ht_to_prev_ht = proportion_to_prev_ht * rel_to_prev_ht
                rel_to_data_ht = (1-proportion_to_prev_ht)*rel_to_prev_ht

                rel_to_xh = dag.layers['new_ht'].rel_z_plus_prop(
                    dag.activations.xh[i],
                    rel_to_data_ht,
                    alpha=alpha, beta=beta
                )

                rel_from_xh_to_ht = rel_to_xh[:, -self.architecture.recur:]
                rel_to_in1 = rel_to_xh[:, :-self.architecture.recur]

                c_i = self._.column_at_a_time * i
                c_j = c_i + self._.column_at_a_time

                rel_to_input[i] = self.dag.layers['input_1'].rel_z_beta_prop(
                    tf.reshape(self.dag.x[:, :, c_i:c_j], shape=[-1, self._.column_at_a_time*self._.dims]),
                    rel_to_in1
                )

                rel_to_prev_ht = rel_from_ht_to_prev_ht + rel_from_xh_to_ht

            pred, heatmaps = self._build_heatmap(sess, x, y, rr_of_pixels=rel_to_input, debug=debug)

        return pred, heatmaps
