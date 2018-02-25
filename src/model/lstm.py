from collections import namedtuple

import tensorflow as tf

from model import base
from model.components.layer import Layer
from utils import experiment_artifact
from utils import logging as lg
from utils import network_architecture, data_provider

lg.set_logging()


Architecture = namedtuple('Architecture', ['size', 'out', 'recur'])


def load(model_path):
    return Network.load(model_path)


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: Architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        self.layers = dict()

        with tf.variable_scope('LSTM') as vs:
            final_output_layer = Layer((architecture.size, architecture.out),
                                       default_biases=tf.Variable(tf.zeros((1, architecture.out))),
                                       name='lstm_final_output')

            self.layers['final_output'] = final_output_layer
            print('No. of variables %d' %
                  (int((dims * no_input_cols + architecture.size) * (4 * architecture.size)) + 4 * architecture.size
                   + architecture.size * architecture.out + architecture.out)
                  )

            lstm = tf.contrib.rnn.BasicLSTMCell(architecture.size)
            lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                                 input_keep_prob=self.keep_prob,
                                                 output_keep_prob=self.keep_prob,
                                                 state_keep_prob=self.keep_prob)
            initialize_memory = tf.zeros([tf.shape(self.x)[0], architecture.size])
            initialize_hidden = tf.zeros([tf.shape(self.x)[0], architecture.size])

            state = initialize_memory, initialize_hidden

            # define  dag
            for i in range(0, max_seq_length, no_input_cols):
                ii = tf.reshape(self.x[:, :, i:i + no_input_cols], [-1, no_input_cols * dims])
                output, state = lstm(ii, state)

            self.y_pred = tf.matmul(output, final_output_layer.W) + final_output_layer.b

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

        raise 'Not implemented'

