import logging
import yaml
import tensorflow as tf
import numpy as np
import inspect


from datetime import datetime
from utils import logging as lg
from utils import data_provider
from utils import network_architecture

from notebook_utils import plot

lg.set_logging()


class Layer:
    def __init__(self, dims, name, stddev=0.1):
        weights = tf.Variable(
            tf.truncated_normal(dims, stddev=stddev),
            name="%s_weights" % name
        )

        bias = tf.Variable(
            tf.zeros(dims[1]),
            name="%s_bias" % name
        )

        self.W = weights
        self.b = bias


class RNNNetwork(object):
    @staticmethod
    def experiment_name():
        return datetime.now().strftime('rnn-%Y-%m-%d--%H-%M')

    def s3_network(self, seq_length=1, epoch=1, lr=0.01, batch=100,
                   architecture_str='in1:_|hidden:_|out1:_|out2:_|--recur:_',
                   verbose=False, result_dir='./experiment-results'):

        experiment_name = self.experiment_name()
        logging.info('Train sprint2 network')
        logging.info('Experiment name : %s' % experiment_name)
        mnist = data_provider.MNISTData()

        # no.rows and cols
        dims, max_seq_length = mnist.train2d.x.shape[1:]
        architecture = network_architecture.parse_architecture(architecture_str)
        logging.info('Network architecture')
        logging.info(architecture)

        no_input_cols = max_seq_length // seq_length
        logging.info('Training %d columns at a time' % no_input_cols)

        # define layers
        ly_input_1 = Layer((dims*no_input_cols, architecture['in1']), 'input_1')
        ly_input_to_cell = Layer((architecture['in1'] + architecture['recur'], architecture['hidden']), 'input_to_cell')

        ly_output_from_cell = Layer((architecture['hidden'], architecture['out1']), 'output_from_cell')
        ly_output_2 = Layer((architecture['out1'], architecture['out2']), 'final_output')

        ly_recurrent = Layer((architecture['hidden'], architecture['recur']), 'recurrent')

        # define placeholders
        rx = tf.placeholder(tf.float32, shape=(None, architecture['recur']), name='recurrent_input')
        x = tf.placeholder(tf.float32, shape=(None, dims, dims), name='data_input')
        y_target = tf.placeholder(tf.float32, [None, 10], name='output_target')

        rr = rx

        # define  dag
        for i in range(0, max_seq_length, no_input_cols):
            ii = tf.reshape(x[:, i:i + no_input_cols], [-1, no_input_cols * dims])
            itc = tf.nn.relu(tf.matmul(ii, ly_input_1.W) + ly_input_1.b)

            xr = tf.concat([itc, rr], axis=1)
            ha = tf.nn.relu(tf.matmul(xr, ly_input_to_cell.W) + ly_input_to_cell.b)

            rr = tf.nn.relu(tf.matmul(ha, ly_recurrent.W) + ly_recurrent.b)

            ho = tf.nn.relu(tf.matmul(ha, ly_output_from_cell.W) + ly_output_from_cell.b)
            ot = tf.matmul(ho, ly_output_2.W) + ly_output_2.b

        y_pred = ot

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_target))
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_op)
        init_op = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(y_target, 1), tf.argmax(y_pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            # todo : save model to some where
            sess.run(init_op)
            step = 1
            for i in range(epoch):
                logging.info('epoch %d' % (i + 1))
                for bx, by in mnist.train2d.get_batch(no_batch=batch):

                    rx0 = np.zeros((batch, architecture['recur']))
                    sess.run(train_op, feed_dict={x: bx, y_target: by, rx: rx0})

                    if (step % 1000 == 0 or step < 10) and verbose:
                        rx0 = np.zeros((len(by), architecture['recur']))
                        acc, loss = sess.run([accuracy, loss_op], feed_dict={x: bx, y_target: by, rx: rx0})
                        logging.info('step %d : current train batch acc %f, loss %f' % (step, acc, loss))

                    step = step + 1

            logging.info('Calculating test accuracy')
            rx0 = np.zeros((len(mnist.test2d.y), architecture['recur']))
            acc = float(sess.run(accuracy, feed_dict={x: mnist.test2d.x, y_target: mnist.test2d.y, rx: rx0}))

            res = dict(
                experiment_name=experiment_name,
                seq_length=seq_length,
                epoch=epoch,
                column_at_a_time=no_input_cols,
                accuracy=acc,
                lr=lr,
                **dict([('layer_%s' % k, v) for k, v in architecture.items()]),
                architecture_name=inspect.currentframe().f_code.co_name
            )

            logging.info('\n%s\n', plot.tabularize_params(res))

            result_path = '%s/%s.yaml' % (result_dir, experiment_name)
            logging.info('Saving result to %s' % result_path)
            with open(result_path, 'w') as outfile:
                yaml.dump(res, outfile, default_flow_style=False)
