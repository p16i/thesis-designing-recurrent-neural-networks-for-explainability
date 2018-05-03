from collections import namedtuple

import tensorflow as tf

from model.architectures import base, deep
from model.components.layer import Layer
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

            ha = tf.nn.relu(tf.matmul(xr, self.ly_input_to_cell.W) - tf.nn.softplus(self.ly_input_to_cell.b))
            self.ha_activations.append(ha)

            rr = tf.nn.relu(tf.matmul(ha, self.ly_recurrent.W) - tf.nn.softplus(self.ly_recurrent.b))
            rr = tf.nn.dropout(rr, keep_prob=self.keep_prob)
            self.rr_activations.append(rr)

        last_hidden_activation = self.ha_activations[-1]
        ha_do = tf.nn.dropout(last_hidden_activation, keep_prob=self.keep_prob)
        last_output_from_cell = tf.nn.relu(tf.matmul(ha_do, self.ly_output_from_cell.W)
                                           - tf.nn.softplus(self.ly_output_from_cell.b))

        self.output_from_cell_activations = [last_output_from_cell]
        self.y_pred = tf.matmul(last_output_from_cell, self.ly_output_2.W)\
                      - tf.nn.softplus(self.ly_output_2.b)

        self.setup_loss_and_opt()


class Network(deep.Network):
    pass
