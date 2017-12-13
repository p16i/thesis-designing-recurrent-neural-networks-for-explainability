import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf

from model import base
from model.components import lrp
from model.components.layer import Layer
from utils import data_provider
from utils import experiment_artifact
from utils import logging as lg
from utils import network_architecture

lg.set_logging()


S3Architecture = namedtuple('S3Architecture', ['in1', 'hidden', 'out1', 'out2', 'recur'])


def load(model_path):
    return S3Network.load(model_path)


class S3NetworkDAG(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture, optimizer):
        super(S3NetworkDAG, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer)

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
        self.output_from_cell_activations = []
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

            rr = tf.nn.relu(tf.matmul(ha, self.ly_recurrent.W) - tf.nn.softplus(self.ly_recurrent.b))
            self.rr_activations.append(rr)

            ha_do = tf.nn.dropout(ha, keep_prob=self.keep_prob)
            ho = tf.nn.relu(tf.matmul(ha_do, self.ly_output_from_cell.W) - tf.nn.softplus(self.ly_output_from_cell.b))
            self.output_from_cell_activations.append(ho)

            ho_do = tf.nn.dropout(ho, keep_prob=self.keep_prob)
            ot = tf.nn.relu(tf.matmul(ho_do, self.ly_output_2.W) - tf.nn.softplus(self.ly_output_2.b))

        self.y_pred = ot

        self.setup_loss_and_opt()

    def no_variables(self):
        return self.ly_input_1.get_no_variables() + self.ly_input_to_cell.get_no_variables() \
            + self.ly_output_from_cell.get_no_variables() + self.ly_output_2.get_no_variables() \
            + self.ly_recurrent.get_no_variables()


class S3Network(base.BaseNetwork):
    def __init__(self, artifact):
        super(S3Network, self).__init__(artifact)

        self.architecture = S3Architecture(**network_architecture.parse(artifact.architecture))
        self.dag = S3NetworkDAG(artifact.column_at_a_time, 28, 28, self.architecture, artifact.optimizer)

        self.experiment_artifact = artifact
        self._ = artifact

    @staticmethod
    def train(seq_length=1, epoch=1, lr=0.01, batch=100, architecture_str='in1:_|hidden:_|out1:_|out2:_|--recur:_',
              keep_prob=0.5, verbose=False, output_dir='./experiment-result', optimizer='AdamOptimizer',
              dataset='mnist'
              ):

        experiment_name = experiment_artifact.get_experiment_name('s3-%s-seq-%d--' % (dataset, seq_length))
        logging.debug('Train sprint3 network')
        logging.debug('Experiment name : %s' % experiment_name)

        data = data_provider.get_data(dataset)

        # no.rows and cols
        dims, max_seq_length = data.train2d.x.shape[1:]
        architecture = S3Architecture(**network_architecture.parse(architecture_str))
        logging.debug('Network architecture')
        logging.debug(architecture)

        logging.debug('Optimizer %s' % optimizer)

        no_input_cols = max_seq_length // seq_length
        logging.debug('Training %d columns at a time' % no_input_cols)

        dag = S3NetworkDAG(no_input_cols, dims, max_seq_length, architecture, optimizer)

        with tf.Session() as sess:
            sess.run(dag.init_op)
            step = 1
            for i in range(epoch):
                logging.debug('epoch %d' % (i + 1))
                for bx, by in data.train2d.get_batch(no_batch=batch):

                    rx0 = np.zeros((batch, architecture.recur))
                    sess.run(dag.train_op,
                             feed_dict={dag.x: bx, dag.y_target: by, dag.rx: rx0, dag.lr: lr, dag.keep_prob: keep_prob})

                    if (step % 1000 == 0 or step < 10) and verbose:
                        rx0 = np.zeros((len(by), architecture.recur))
                        acc, loss = sess.run([dag.accuracy, dag.loss_op],
                                             feed_dict={dag.x: bx, dag.y_target: by, dag.rx: rx0, dag.keep_prob: 1})

                        rx0 = np.zeros((len(data.val2d.y), architecture.recur))
                        acc_val = sess.run(dag.accuracy, feed_dict={dag.x: data.val2d.x, dag.y_target: data.val2d.y,
                                                                    dag.rx: rx0, dag.keep_prob: 1})
                        logging.debug('step %d : current train batch acc %f, loss %f | val acc %f'
                                     % (step, acc, loss, acc_val))

                    step = step + 1

            logging.debug('Calculating test accuracy')
            rx0 = np.zeros((len(data.test2d.y), architecture.recur))
            acc = float(sess.run(dag.accuracy,
                                 feed_dict={dag.x: data.test2d.x, dag.y_target: data.test2d.y,
                                            dag.rx: rx0, dag.keep_prob: 1}))

            rx0 = np.zeros((len(data.val2d.y), architecture.recur))
            val_acc = float(sess.run(dag.accuracy, feed_dict={dag.x: data.val2d.x, dag.y_target: data.val2d.y,
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
                architecture_name='s3_network',
                dims=dims,
                max_seq_length=max_seq_length,
                keep_prob=keep_prob,
                optimizer=optimizer,
                val_accuracy=val_acc,
                dataset=dataset
            )

            logging.debug('\n%s\n', lg.tabularize_params(res))

            output_dir = '%s/%s' % (output_dir, experiment_name)

            return experiment_artifact.save_artifact(sess, res, output_dir=output_dir)

    def lrp(self, x, factor=1, debug=False):

        x_3d = x.reshape(-1, 28, 28)
        with self.get_session() as sess:

            layer_keys = ['input_1', 'input_to_cell', 'output_from_cell', 'output_2', 'recurrent']
            layer_weights = sess.run([self.dag.layers[k].W for k in layer_keys])
            weights = dict(zip(layer_keys, layer_weights))

            rx = np.zeros((x_3d.shape[0], self.architecture.recur))
            data = sess.run(
                self.dag.input_to_cell_activations + self.dag.ha_activations
                + self.dag.output_from_cell_activations
                + [self.dag.y_pred],
                feed_dict={self.dag.x: x_3d, self.dag.rx: rx, self.dag.keep_prob: 1})

            input_to_cell_activations = np.array(data[:self._.seq_length]).transpose([1, 2, 0])
            ha_activations = np.array(data[self._.seq_length:self._.seq_length*2]).transpose([1, 2, 0])
            output_from_cell_activations = np.array(data[self._.seq_length*2:self._.seq_length*3]).transpose([1, 2, 0])

            pred = data[-1]
            mark = np.zeros(pred.shape)
            mark[range(pred.shape[0]), np.argmax(pred, axis=1)] = 1
            relevance = pred * mark

            dims = self.experiment_artifact.dims

            RR_of_hiddens = np.zeros((x_3d.shape[0], self.architecture.hidden, self._.seq_length))
            RR_of_input1 = np.zeros((x_3d.shape[0], self.architecture.in1, self._.seq_length))
            RR_of_pixels = np.zeros((x_3d.shape[0], dims * self.experiment_artifact.column_at_a_time, self._.seq_length))
            RR_of_rr = np.zeros((x_3d.shape[0], self.architecture.recur, self._.seq_length+1))

            # lwr start here
            RR_of_output_from_cell = lrp.z_plus_prop(
                output_from_cell_activations[:, :, -1], weights['output_2'], relevance, factor=factor)
            RR_of_hiddens[:, :, -1] = lrp.z_plus_prop(ha_activations[:, :, -1], weights['output_from_cell'],
                                                      RR_of_output_from_cell, factor=factor)

            temp = lrp.z_plus_prop(input_to_cell_activations[:, :, -1]
                                   , weights['input_to_cell'], RR_of_hiddens[:, :, -1], factor=factor)
            # temp = np.squeeze(temp)

            RR_of_input1[:, :, -1] = temp[:, :-self.architecture.recur]
            RR_of_rr[:, :, -2] = temp[:, -self.architecture.recur:]

            RR_of_pixels[:, :, -1] = lrp.z_beta_prop(
                x_3d[:, :, -self.experiment_artifact.column_at_a_time:].reshape(x_3d.shape[0], -1),
                weights['input_1'], RR_of_input1[:, :, -1], factor=factor
            )

            for i in range(self._.seq_length - 1)[::-1]:
                RR_of_hiddens[:, :, i] = lrp.z_plus_prop(ha_activations[:, :, i],
                                                         weights['recurrent'], RR_of_rr[:, :, i + 1], factor=factor)

                temp = lrp.z_plus_prop(input_to_cell_activations[:, :, i],
                                       weights['input_to_cell'], RR_of_hiddens[:, :, i], factor=factor)

                RR_of_input1[:, :, i] = temp[:, :-self.architecture.recur]
                RR_of_rr[:, :, i] = temp[:, -self.architecture.recur:]

                c_i = self._.column_at_a_time * i
                c_j = c_i + self._.column_at_a_time

                RR_of_pixels[:, :, i] = lrp.z_beta_prop(
                    x_3d[:, :, c_i:c_j].reshape(x_3d.shape[0], -1),
                    weights['input_1'], RR_of_input1[:, :, i], factor=factor
                )

            if debug:

                logging.debug('Prediction before softmax \n%s' % list(zip(mark, pred)))
                logging.debug('Relevance')
                logging.debug(relevance)
                total_relevance = np.sum(relevance, axis=1)
                logging.debug(total_relevance)

                logging.debug('RR_of_ha')
                logging.debug(RR_of_hiddens.shape)

                total_relevance_hidden_units = np.sum(RR_of_hiddens, axis=1)
                logging.debug(total_relevance_hidden_units)
                logging.debug(np.sum(total_relevance_hidden_units[:, :-1], axis=1))

                logging.debug('RR_of_pixels')
                total_relevance_pixels = np.sum(RR_of_pixels, axis=(1, 2))
                logging.debug(total_relevance_pixels)

                np.testing.assert_allclose(total_relevance_pixels, total_relevance,
                                           rtol=1e-6, atol=0,
                                           err_msg='Conservation property isn`t hold\n'
                                                   ': Sum of relevance from pixels is not equal to output relevance.')

        heatmaps = np.zeros(x_3d.shape)
        for i in range(0, heatmaps.shape[2], self._.column_at_a_time):
            t_idx = int(i / self._.column_at_a_time)
            heatmaps[:, :, i:(i + self._.column_at_a_time)] = RR_of_pixels[:, :, t_idx] \
                .reshape(heatmaps.shape[0], heatmaps.shape[1], -1)

        # max value in a row
        return np.argmax(pred, axis=1), heatmaps

