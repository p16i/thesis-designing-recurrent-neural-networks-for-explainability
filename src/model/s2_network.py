from collections import namedtuple

import tensorflow as tf

from model import base
from model.components.layer import Layer
from utils import experiment_artifact
from utils import logging as lg
from utils import network_architecture, data_provider

lg.set_logging()


Architecture = namedtuple('S2Architecture', ['hidden', 'out', 'recur'])


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: Architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        self.ly_input = Layer((dims*no_input_cols + architecture.recur, architecture.hidden), 's2__input')

        self.ly_output = Layer((architecture.hidden, architecture.out), 's2__output')

        self.ly_recurrent = Layer((architecture.hidden, architecture.recur), 's2__recurrent')

        self.layers = {
            'input': self.ly_input,
            'output': self.ly_output,
            'recurrent': self.ly_recurrent
        }

        rr = self.rx

        self.ha_activations = []
        self.rr_activations = [self.rx]

        # define  dag
        for i in range(0, max_seq_length, no_input_cols):
            ii = tf.reshape(self.x[:, :, i:i + no_input_cols], [-1, no_input_cols * dims])
            xr = tf.concat([ii, rr], axis=1)

            ha = tf.nn.relu(tf.matmul(xr, self.ly_input.W) - tf.nn.softplus(self.ly_input.b))
            self.ha_activations.append(ha)

            ha_do = tf.nn.dropout(ha, keep_prob=self.keep_prob)
            rr = tf.nn.relu(tf.matmul(ha_do, self.ly_recurrent.W) - tf.nn.softplus(self.ly_recurrent.b))
            self.rr_activations.append(rr)

        self.y_pred = tf.matmul(self.ha_activations[-1], self.ly_output.W) - tf.nn.softplus(self.ly_output.b)

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

        self.name = 's2_network'

    def lrp(self, x, y, alpha=1.0, beta=0.0, debug=False):

        x_3d = x.reshape(-1, self.data_no_rows, self.data_no_cols)
        with self.get_session() as sess:

            # lwr start here
            self.dag.setup_variables_for_lrp()
            rel_to_input = [None]*self._.seq_length

            rel_to_hidden = self.dag.layers['output'].rel_z_plus_prop(
                self.dag.ha_activations[-1],
                self.dag.total_relevance, beta=beta, alpha=alpha
            )
            weight_px_parts = self.dag.layers['input'].W[:-self.architecture.recur, :]
            weight_rr_parts = self.dag.layers['input'].W[-self.architecture.recur:, :]

            rel_to_recurrent, rel_to_input[-1] = Layer.rel_z_plus_beta_prop(
                self.dag.rr_activations[-2],
                weight_rr_parts,
                tf.reshape(self.dag.x[:, :, -self.experiment_artifact.column_at_a_time:], shape=[x_3d.shape[0], -1]),
                weight_px_parts,
                rel_to_hidden,
                beta=beta,
                alpha=alpha
            )

            for i in range(self._.seq_length - 1)[::-1]:
                rel_to_hidden = self.dag.layers['recurrent'].rel_z_plus_prop(
                    self.dag.ha_activations[i],
                    rel_to_recurrent, beta=beta, alpha=alpha
                )

                c_i = self._.column_at_a_time * i
                c_j = c_i + self._.column_at_a_time

                rel_to_recurrent, rel_to_input[i] = Layer.rel_z_plus_beta_prop(
                    self.dag.rr_activations[i],
                    weight_rr_parts,
                    tf.reshape(self.dag.x[:, :, c_i:c_j], shape=[x_3d.shape[0], -1]),
                    weight_px_parts,
                    rel_to_hidden,
                    beta=beta, alpha=alpha
                )

            pred, heatmaps = self._build_heatmap(sess, x, y,
                                                 rr_of_pixels=rel_to_input, debug=debug)

        return pred, heatmaps
