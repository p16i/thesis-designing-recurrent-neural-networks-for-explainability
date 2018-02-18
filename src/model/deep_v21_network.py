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


Architecture = namedtuple('S3Architecture', ['in1', 'hidden', 'out1', 'out2', 'recur'])


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        # define layers
        self.ly_input_1 = Layer((dims*no_input_cols, architecture.in1), 's3__input_1')
        self.ly_input_to_cell = Layer((architecture.in1 + architecture.recur, architecture.hidden), 's3__input_to_cell')

        self.ly_output_from_cell = Layer((architecture.hidden, architecture.out1), 's3__output_from_cell')
        self.ly_output_2 = Layer((architecture.out1, architecture.out2), 's3__final_output')

        self.ly_recurrent = Layer((architecture.hidden, architecture.recur), 's3__recurrent')

        self.layers = {
            'input_1': self.ly_input_1,
            'input_to_cell': self.ly_input_to_cell,
            'output_from_cell': self.ly_output_from_cell,
            'output_2': self.ly_output_2,
            'recurrent': self.ly_recurrent
        }

        rr = self.rx

        self.ha_activations = []
        self.hidden_to_recur_activations = []

        self.rr_activations = [self.rx]
        self.input_1_activations = []
        self.input_to_cell_activations = []
        # define  dag
        for i in range(0, max_seq_length, no_input_cols):
            ii = tf.reshape(self.x[:, :, i:i + no_input_cols], [-1, no_input_cols * dims])
            itc = tf.nn.relu(tf.matmul(ii, self.ly_input_1.W) - tf.nn.softplus(self.ly_input_1.b))
            self.input_1_activations.append(itc)

            itc_do = tf.nn.dropout(itc, keep_prob=self.keep_prob)

            xr = tf.concat([itc_do, rr], axis=1)
            self.input_to_cell_activations.append(xr)
            xr_do = tf.nn.dropout(xr, keep_prob=self.keep_prob)

            ha = tf.nn.relu(tf.matmul(xr_do, self.ly_input_to_cell.W) - tf.nn.softplus(self.ly_input_to_cell.b))
            self.ha_activations.append(ha)

            ha_do = tf.nn.dropout(ha, keep_prob=self.keep_prob)
            rr_from_hidden = tf.nn.relu(tf.matmul(ha_do, self.ly_recurrent.W) - tf.nn.softplus(self.ly_recurrent.b))

            ha_from_cell = tf.nn.relu(tf.matmul(ha_do, self.ly_output_from_cell.W)
                                           - tf.nn.softplus(self.ly_output_from_cell.b))

            rr = rr_from_hidden * ha_from_cell
            self.rr_activations.append(rr)

        last_hidden_activation = self.ha_activations[-1]
        ha_do = tf.nn.dropout(last_hidden_activation, keep_prob=self.keep_prob)
        last_output_from_cell = tf.nn.relu(tf.matmul(ha_do, self.ly_output_from_cell.W)
                                           - tf.nn.softplus(self.ly_output_from_cell.b))

        self.output_from_cell_activations = [last_output_from_cell]
        self.y_pred = tf.matmul(last_output_from_cell, self.ly_output_2.W)\
                      - tf.nn.softplus(self.ly_output_2.b)

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
            rel_to_input = [None]*self._.seq_length

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

