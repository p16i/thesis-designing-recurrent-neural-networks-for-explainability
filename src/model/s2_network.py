import logging
import tensorflow as tf
import numpy as np
from collections import namedtuple

import lwr
from rnn_network import Layer
from utils import logging as lg
from utils import data_provider
from utils import network_architecture

from notebook_utils import plot

from utils import experiment_artifact

from model import base

lg.set_logging()


S2Architecture = namedtuple('S2Architecture', ['hidden', 'out', 'recur'])


def load(model_path):
    return S2Network.load(model_path)


class S2NetworkDAG(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: S2Architecture, optimizer):
        super(S2NetworkDAG, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer)

        self.ly_input = Layer((dims*no_input_cols + architecture.recur, architecture.hidden), 's2__input')

        self.ly_output = Layer((architecture.hidden, architecture.out), 's2__output')

        self.ly_recurrent = Layer((architecture.hidden, architecture.recur), 's2__recurrent')

        self.layers = {
            'input': self.ly_input,
            'output': self.ly_output,
            'recurrent': self.ly_recurrent
        }

        # define placeholders
        self.rx = tf.placeholder(tf.float32, shape=(None, architecture.recur), name='s2__recurrent_input')
        self.x = tf.placeholder(tf.float32, shape=(None, dims, dims), name='s2__data_input')
        self.y_target = tf.placeholder(tf.float32, [None, 10], name='s2__output_target')
        self.lr = tf.placeholder(tf.float32, name='s2__lr')
        self.keep_prob = tf.placeholder(tf.float32, name='s2__keep_prob')

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

            ot = tf.nn.relu(tf.matmul(ha, self.ly_output.W) - tf.nn.softplus(self.ly_output.b))

        self.y_pred = ot

        self.setup_loss_and_opt()

    def no_variables(self):
        return self.ly_input.get_no_variables() + self.ly_output.get_no_variables() + self.ly_recurrent.get_no_variables()


class S2Network(base.BaseNetwork):
    def __init__(self, artifact: experiment_artifact.Artifact):
        super(S2Network, self).__init__(artifact)

        self.architecture = S2Architecture(**network_architecture.parse(artifact.architecture))

        tf.reset_default_graph()
        self.dag = S2NetworkDAG(artifact.column_at_a_time, 28, 28, self.architecture, artifact.optimizer)

        self.experiment_artifact = artifact
        self._ = artifact

    @staticmethod
    def train(seq_length=1, epoch=1, lr=0.01, batch=100, keep_prob=0.5, architecture_str='hidden:_|out:_|--recur:_',
              verbose=False, output_dir='./experiment-result', optimizer='AdamOptimizer'
              ):

        experiment_name = experiment_artifact.get_experiment_name()
        logging.debug('Train s2 network')
        logging.debug('Experiment name : %s' % experiment_name)
        mnist = data_provider.MNISTData()

        # no.rows and cols
        dims, max_seq_length = mnist.train2d.x.shape[1:]
        architecture = S2Architecture(**network_architecture.parse(architecture_str))
        logging.debug('Network architecture')
        logging.debug(architecture)

        no_input_cols = max_seq_length // seq_length
        logging.debug('Training %d columns at a time' % no_input_cols)
        logging.debug('Optimizer %s' % optimizer)

        dag = S2NetworkDAG(no_input_cols, dims, max_seq_length, architecture, optimizer)

        with tf.Session() as sess:
            sess.run(dag.init_op)
            step = 1
            for i in range(epoch):
                logging.debug('epoch %d' % (i + 1))
                for bx, by in mnist.train2d.get_batch(no_batch=batch):

                    rx0 = np.zeros((batch, architecture.recur))
                    sess.run(dag.train_op,
                             feed_dict={dag.x: bx, dag.y_target: by, dag.rx: rx0, dag.lr: lr, dag.keep_prob: keep_prob})

                    if (step % 1000 == 0 or step < 10) and verbose:
                        rx0 = np.zeros((len(by), architecture.recur))
                        acc, loss = sess.run([dag.accuracy, dag.loss_op],
                                             feed_dict={dag.x: bx, dag.y_target: by, dag.rx: rx0, dag.keep_prob: 1})

                        rx0 = np.zeros((len(mnist.val2d.y), architecture.recur))
                        acc_val = sess.run(dag.accuracy, feed_dict={dag.x: mnist.val2d.x, dag.y_target: mnist.val2d.y,
                                                                    dag.rx: rx0, dag.keep_prob: 1})
                        logging.debug('step %d : current train batch acc %f, loss %f | val acc %f'
                                     % (step, acc, loss, acc_val))

                    step = step + 1

            rx0 = np.zeros((len(mnist.val2d.y), architecture.recur))
            acc_val = sess.run(dag.accuracy, feed_dict={dag.x: mnist.val2d.x, dag.y_target: mnist.val2d.y,
                                                        dag.rx: rx0, dag.keep_prob: 1})
            logging.debug('Val accuracy : %f' % acc_val)

            logging.debug('Calculating test accuracy')
            rx0 = np.zeros((len(mnist.test2d.y), architecture.recur))
            acc = float(sess.run(dag.accuracy,
                                 feed_dict={dag.x: mnist.test2d.x, dag.y_target: mnist.test2d.y,
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
                architecture_name='s2_network',
                dims=dims,
                max_seq_length=max_seq_length,
                keep_prob=keep_prob,
                optimizer=optimizer
            )

            logging.debug('\n%s\n', plot.tabularize_params(res))

            output_dir = '%s/%s' % (output_dir, experiment_name)

            experiment_artifact.save_artifact(sess, res, output_dir=output_dir)

    def lwr(self, x, debug=False):

        x_3d = x.reshape(-1, 28, 28)
        with self.get_session() as sess:

            layer_keys = ['input', 'output', 'recurrent']
            layer_weights = sess.run([self.dag.layers[k].W for k in layer_keys])
            weights = dict(zip(layer_keys, layer_weights))

            rx = np.zeros((x_3d.shape[0], self.architecture.recur))
            pred = sess.run([self.dag.y_pred], feed_dict={self.dag.x: x_3d, self.dag.rx: rx, self.dag.keep_prob: 1})[0]
            mark = np.zeros(pred.shape)
            mark[range(pred.shape[0]), np.argmax(pred, axis=1)] = 1

            relevance = pred * mark

            data = sess.run(self.dag.ha_activations + self.dag.rr_activations,
                            feed_dict={self.dag.x: x_3d, self.dag.rx: rx, self.dag.keep_prob: 1})
            ha_activations = np.array(data[:self._.seq_length]).transpose([1, 2, 0])
            rr_activations = np.array(data[self._.seq_length:]).transpose([1, 2, 0])

            dims = self.experiment_artifact.dims

            RR_of_hiddens = np.zeros(
                (x_3d.shape[0], self.architecture.hidden, self._.seq_length))
            RR_of_pixels = np.zeros((x_3d.shape[0], dims * self.experiment_artifact.column_at_a_time, self._.seq_length))
            RR_of_rr = np.zeros((x_3d.shape[0], self.architecture.recur, self._.seq_length+1))

            # lwr start here
            RR_of_hiddens[:, :, -1] = lwr.z_plus_prop(ha_activations[:, :, -1], weights['output'], relevance)

            weight_px_parts = weights['input'][:-self.architecture.recur, :]
            weight_rr_parts = weights['input'][-self.architecture.recur:, :]
            RR_of_rr[:, :, -2], RR_of_pixels[:, :, -1] = lwr.z_plus_beta_prop(
                rr_activations[:, :, -2],
                weight_rr_parts,
                x_3d[:, :, -self.experiment_artifact.column_at_a_time:].reshape(x_3d.shape[0], -1),
                weight_px_parts,
                RR_of_hiddens[:, :, -1]
            )

            for i in range(self._.seq_length - 1)[::-1]:
                RR_of_hiddens[:, :, i] = lwr.z_plus_prop(ha_activations[:, :, i], weights['recurrent'], RR_of_rr[:, :, i + 1])

                c_i = self._.column_at_a_time * i
                c_j = c_i + self._.column_at_a_time

                RR_of_rr[:, :, i], RR_of_pixels[:, :, i] = lwr.z_plus_beta_prop(
                    rr_activations[:, :, i],
                    weight_rr_parts,
                    x_3d[:, :, c_i:c_j].reshape(x_3d.shape[0], -1),
                    weight_px_parts,
                    RR_of_hiddens[:, :, i]
                )

            if debug:
                logging.debug('Prediction before softmax \n%s' % list(zip(mark,pred)))
                logging.debug('Relevance %f' % np.sum(relevance))
                logging.debug('RR_of_ha')
                logging.debug(np.sum(RR_of_hiddens, axis=0))

                logging.debug('RR_of_rr + pixels')
                sum_px_rr = np.sum(RR_of_rr[:, :-1], axis=0) + np.sum(RR_of_pixels, axis=0)
                logging.debug(sum_px_rr)
                logging.debug('----------')
                logging.debug('RR_of_pixels')
                logging.debug(np.sum(RR_of_pixels, axis=0))
                logging.debug('RR_of_rr')
                logging.debug(np.sum(RR_of_rr, axis=0))
                logging.debug('============')
                logging.debug('Total Relevance of input pixels %f', np.sum(RR_of_pixels))

        heatmaps = np.zeros(x_3d.shape)
        for i in range(0, heatmaps.shape[2], self._.column_at_a_time):
            t_idx = int(i / self._.column_at_a_time)
            heatmaps[:, :, i:(i + self._.column_at_a_time)] = RR_of_pixels[:, :, t_idx]\
                .reshape(heatmaps.shape[0], heatmaps.shape[1], -1)

        # max value in a row
        return np.argmax(pred, axis=1), heatmaps
