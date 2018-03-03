from collections import namedtuple

import tensorflow as tf

from model.architectures import base
from model.components.layer import Layer
from utils import experiment_artifact
from utils import logging as lg
from utils import network_architecture

lg.set_logging()


Architecture = namedtuple('Architecture', ['hidden_l1', 'out_l1', 'recur', 'hidden_l2', 'out', 'recur_l2'])


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: Architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        self.ly_input_l1 = Layer((dims*no_input_cols + architecture.recur, architecture.hidden_l1),
                                 'shallow_2_levels__input_l1')

        self.ly_output_l1 = Layer((architecture.hidden_l1, architecture.out_l1), 'shallow_2_levels__output_l2')

        self.ly_recurrent_l1 = Layer((architecture.hidden_l1, architecture.recur), 'shallow_2_levels__recurrent_l1')

        self.ly_input_l2 = Layer((architecture.recur_l2+architecture.out_l1, architecture.hidden_l2),
                                 'shallow_2_levels__input_l2')

        self.ly_output_l2 = Layer((architecture.hidden_l2, architecture.out), 'shallow_2_levels__output_l2')
        self.ly_recurrent_l2 = Layer((architecture.hidden_l2, architecture.recur_l2), 'shallow_2_levels__recurrent_l2')

        self.layers = {
            'input_l1': self.ly_input_l1,
            'output_l1': self.ly_output_l1,
            'recurrent_l1': self.ly_recurrent_l1,
            'input_l2': self.ly_input_l2,
            'output_l2': self.ly_output_l2,
            'recurrent_l2': self.ly_recurrent_l2
        }

        self.rr_l2 = tf.zeros((tf.shape(self.x)[0], architecture.recur_l2))

        rr = self.rx
        rr_l2 = self.rr_l2

        self.input_l2_activations = []
        self.ha_l1_activations = []
        self.ha_l2_activations = []

        self.rr_l1_activations = [self.rx]
        self.rr_l2_activations = [rr_l2]

        # define  dag
        for i in range(0, max_seq_length, no_input_cols):
            # level 1
            ii = tf.reshape(self.x[:, :, i:i + no_input_cols], [-1, no_input_cols * dims])
            xr = tf.concat([ii, rr], axis=1)

            ha = tf.nn.relu(tf.matmul(xr, self.ly_input_l1.W) - tf.nn.softplus(self.ly_input_l1.b))
            self.ha_l1_activations.append(ha)

            oo_l1 = tf.nn.dropout(tf.matmul(ha, self.ly_output_l1.W) - tf.nn.softplus(self.ly_output_l1.b),
                                  keep_prob=self.keep_prob)

            rr = tf.nn.relu(tf.matmul(ha, self.ly_recurrent_l1.W) - tf.nn.softplus(self.ly_recurrent_l1.b))
            self.rr_l1_activations.append(rr)

            # level 2
            xr_l2 = tf.concat([oo_l1, rr_l2], axis=1)
            self.input_l2_activations.append(xr_l2)

            ha_l2 = tf.nn.relu(tf.matmul(xr_l2, self.ly_input_l2.W) - tf.nn.softplus(self.ly_input_l2.b))
            self.ha_l2_activations.append(ha_l2)

            rr_l2 = tf.nn.relu(tf.matmul(ha_l2, self.ly_recurrent_l2.W) - tf.nn.softplus(self.ly_recurrent_l2.b))
            self.rr_l2_activations.append(rr_l2)

        self.y_pred = tf.matmul(self.ha_l2_activations[-1], self.ly_output_l2.W) - tf.nn.softplus(self.ly_output_l2.b)

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

        self.name = 'shallow_2_levels'

    def lrp(self, x, y, alpha=1.0, beta=0.0, debug=False):

        x_3d = x.reshape(-1, self.data_no_rows, self.data_no_cols)
        with self.get_session() as sess:

            # lwr start here
            self.dag.setup_variables_for_lrp()
            rel_to_input = [None]*self._.seq_length

            rel_to_hidden_l2 = self.dag.layers['output_l2'].rel_z_plus_prop(
                self.dag.ha_l2_activations[-1],
                self.dag.total_relevance, beta=beta, alpha=alpha
            )

            rel_to_input_l2 = self.dag.layers['input_l2'].rel_z_plus_prop(
                self.dag.input_l2_activations[-1],
                rel_to_hidden_l2, beta=beta, alpha=alpha
            )

            rel_to_output_l1 = rel_to_input_l2[:, :-self.architecture.recur_l2]
            rel_to_rr_l2 = rel_to_input_l2[:, -self.architecture.recur_l2:]


            rel_to_hidden_l1 = self.dag.layers['output_l1'].rel_z_plus_prop(
                self.dag.ha_l1_activations[-1],
                rel_to_output_l1,
                alpha=alpha, beta=beta)


            weight_px_parts = self.dag.layers['input_l1'].W[:-self.architecture.recur, :]
            weight_rr_parts = self.dag.layers['input_l1'].W[-self.architecture.recur:, :]

            rel_to_rr_l1, rel_to_input[-1] = Layer.rel_z_plus_beta_prop(
                self.dag.rr_l1_activations[-2],
                weight_rr_parts,
                tf.reshape(self.dag.x[:, :, -self.experiment_artifact.column_at_a_time:], shape=[x_3d.shape[0], -1]),
                weight_px_parts,
                rel_to_hidden_l1,
                alpha=alpha, beta=beta
            )


            for i in range(self._.seq_length - 1)[::-1]:
                # level 2
                rel_to_hidden_l2 = self.dag.layers['recurrent_l2'].rel_z_plus_prop(
                    self.dag.ha_l2_activations[i],
                    rel_to_rr_l2,
                    alpha=alpha, beta=beta
                )


                rel_to_input_l2 = self.dag.layers['input_l2'].rel_z_plus_prop(
                    self.dag.input_l2_activations[i],
                    rel_to_hidden_l2, beta=beta, alpha=alpha
                )

                rel_to_output_l1 = rel_to_input_l2[:, :-self.architecture.recur_l2]
                rel_to_rr_l2 = rel_to_input_l2[:, -self.architecture.recur_l2:]

                # level 1
                rel_to_hidden_l1_from_output1 = self.dag.layers['output_l1'].rel_z_plus_prop(
                    self.dag.ha_l1_activations[i],
                    rel_to_output_l1,
                    alpha=alpha, beta=beta)

                rel_to_hidden_l1_from_rr_l1 = self.dag.layers['recurrent_l1'].rel_z_plus_prop(
                    self.dag.ha_l1_activations[i],
                    rel_to_rr_l1,
                    alpha=alpha, beta=beta
                )

                rel_to_hidden_l1 = rel_to_hidden_l1_from_output1 + rel_to_hidden_l1_from_rr_l1

                c_i = self._.column_at_a_time * i
                c_j = c_i + self._.column_at_a_time

                rel_to_rr_l1, rel_to_input[i] = Layer.rel_z_plus_beta_prop(
                    self.dag.rr_l1_activations[i],
                    weight_rr_parts,
                    tf.reshape(self.dag.x[:, :, c_i:c_j], shape=[x_3d.shape[0], -1]),
                    weight_px_parts,
                    rel_to_hidden_l1,
                    beta=beta, alpha=alpha
                )

            pred, heatmaps = self._build_heatmap(sess, x, y,
                                                 rr_of_pixels=rel_to_input, debug=debug)

        return pred, heatmaps
