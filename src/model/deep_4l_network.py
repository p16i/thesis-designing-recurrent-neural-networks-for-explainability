import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf

from model import base
from model.components.layer import Layer
from utils import data_provider
from utils import experiment_artifact
from utils import logging as lg
from utils import network_architecture

lg.set_logging()

Architecture = namedtuple('Deep4LArchitecture', ['in1', 'in2', 'hidden', 'out1', 'out2', 'recur'])

ARCHITECTURE_NAME = 'deep_4l_network'


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: Architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        # define layers
        self.ly_input_1 = Layer((dims*no_input_cols, architecture.in1), 'deep_4l__input_1')
        self.ly_input_2 = Layer((architecture.in1, architecture.in2), 'deep_4l__input_2')
        self.ly_input_to_cell = Layer((architecture.in2 + architecture.recur, architecture.hidden), 'deep_4l__input_to_cell')

        self.ly_output_from_cell = Layer((architecture.hidden, architecture.out1), 'deep_4l__output_from_cell')
        self.ly_output_2 = Layer((architecture.out1, architecture.out2), 'deep_4l__final_output')

        self.ly_recurrent = Layer((architecture.hidden, architecture.recur), 'deep_4l__recurrent')

        self.layers = {
            'input_1': self.ly_input_1,
            'input_2': self.ly_input_2,
            'input_to_cell': self.ly_input_to_cell,
            'output_from_cell': self.ly_output_from_cell,
            'output_2': self.ly_output_2,
            'recurrent': self.ly_recurrent
        }

        rr = self.rx

        self.ha_activations = []
        self.rr_activations = [self.rx]
        self.input_1_activations = []
        self.input_2_activations = []
        self.input_to_cell_activations = []
        self.output_from_cell_activations = []
        # define  dag
        for i in range(0, max_seq_length, no_input_cols):
            ii = tf.reshape(self.x[:, :, i:i + no_input_cols], [-1, no_input_cols * dims])
            in1 = tf.nn.relu(tf.matmul(ii, self.ly_input_1.W) - tf.nn.softplus(self.ly_input_1.b))
            self.input_1_activations.append(in1)
            in1_do = tf.nn.dropout(in1, keep_prob=self.keep_prob)

            in2 = tf.nn.relu(tf.matmul(in1_do, self.ly_input_2.W) - tf.nn.softplus(self.ly_input_2.b))
            self.input_2_activations.append(in2)
            in2_do = tf.nn.dropout(in2, keep_prob=self.keep_prob)

            xr = tf.concat([in2_do, rr], axis=1)
            self.input_to_cell_activations.append(xr)
            xr_do = tf.nn.dropout(xr, keep_prob=self.keep_prob)

            ha = tf.nn.relu(tf.matmul(xr_do, self.ly_input_to_cell.W) - tf.nn.softplus(self.ly_input_to_cell.b))
            self.ha_activations.append(ha)

            rr = tf.nn.relu(tf.matmul(ha, self.ly_recurrent.W) - tf.nn.softplus(self.ly_recurrent.b))
            self.rr_activations.append(rr)

            ha_do = tf.nn.dropout(ha, keep_prob=self.keep_prob)
            ho = tf.nn.relu(tf.matmul(ha_do, self.ly_output_from_cell.W) - tf.nn.softplus(self.ly_output_from_cell.b))
            self.output_from_cell_activations.append(ho)

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

    def lrp(self, x, factor=1, debug=False):

        x_3d = x.reshape(-1, 28, 28)
        with self.get_session() as sess:
            # lwr start here
            self.dag.setup_variables_for_lrp()
            rel_to_input = [None]*self._.seq_length

            # lwr start here
            rel_to_output_from_cell = self.dag.layers['output_2'].rel_z_plus_prop(
                self.dag.output_from_cell_activations[-1],
                self.dag.total_relevance, factor=factor
            )

            rel_to_hidden = self.dag.layers['output_from_cell'].rel_z_plus_prop(
                self.dag.ha_activations[-1],
                rel_to_output_from_cell, factor=factor
            )

            rel_to_input_to_cell = self.dag.layers['input_to_cell'].rel_z_plus_prop(
                self.dag.input_to_cell_activations[-1],
                rel_to_hidden, factor=factor
            )

            rel_to_recurrent = rel_to_input_to_cell[:, -self.architecture.recur:]

            rel_from_hidden_to_in2 = rel_to_input_to_cell[:, :-self.architecture.recur]

            rel_to_in1 = self.dag.layers['input_2'].rel_z_plus_prop(
                self.dag.input_1_activations[-1],
                rel_from_hidden_to_in2, factor=factor

            )

            rel_to_input[-1] = self.dag.layers['input_1'].rel_z_beta_prop(
                tf.reshape(self.dag.x[:, :, -self.experiment_artifact.column_at_a_time:], shape=[x_3d.shape[0], -1]),
                rel_to_in1
            )

            for i in range(self._.seq_length - 1)[::-1]:
                rel_to_hidden = self.dag.layers['recurrent'].rel_z_plus_prop(
                    self.dag.ha_activations[i],
                    rel_to_recurrent, factor=factor
                )

                rel_to_input_to_cell = self.dag.layers['input_to_cell'].rel_z_plus_prop(
                    self.dag.input_to_cell_activations[i],
                    rel_to_hidden, factor=factor
                )

                rel_to_recurrent = rel_to_input_to_cell[:, -self.architecture.recur:]

                rel_from_hidden_to_in2 = rel_to_input_to_cell[:, :-self.architecture.recur]

                rel_to_in1 = self.dag.layers['input_2'].rel_z_plus_prop(
                    self.dag.input_1_activations[i],
                    rel_from_hidden_to_in2, factor=factor

                )

                c_i = self._.column_at_a_time * i
                c_j = c_i + self._.column_at_a_time

                rel_to_input[i] = self.dag.layers['input_1'].rel_z_beta_prop(
                    tf.reshape(self.dag.x[:, :, c_i:c_j], shape=[x_3d.shape[0], -1]),
                    rel_to_in1
                )

            pred, heatmaps = self._build_heatmap(sess, x,
                                                 rr_of_pixels=rel_to_input, debug=debug)

        return pred, heatmaps
