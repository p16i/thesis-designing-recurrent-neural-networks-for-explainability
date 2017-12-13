import tensorflow as tf
import logging
import numpy as np
from utils import logging as lg

from utils import experiment_artifact
lg.set_logging()


class BaseDag:
    def __init__(self, architecture, dims, max_seq_length, optimizer):
        tf.reset_default_graph()

        self.rx = tf.placeholder(tf.float32, shape=(None, architecture.recur), name='recurrent_input')
        self.x = tf.placeholder(tf.float32, shape=(None, dims, max_seq_length), name='input')
        self.y_target = tf.placeholder(tf.float32, [None, 10], name='output_target')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.loss_op = None
        self.train_op = None
        self.init_op = None
        self.accuracy = None

        self.layers = []

        self.optimizer = optimizer

    def setup_loss_and_opt(self):
        self.loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_target)
        )

        optimizer = getattr(tf.train, self.optimizer)
        self.train_op = optimizer(learning_rate=self.lr).minimize(self.loss_op)
        self.init_op = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(self.y_target, 1), tf.argmax(self.y_pred, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


class BaseNetwork:
    def __init__(self, artifact: experiment_artifact.Artifact):

        self.experiment_artifact = artifact
        self._ = artifact
        tf.reset_default_graph()

    def get_session(self):

        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, '%s/model.ckpt' % self._.path)

        return sess

    def compute_grad_wrt_x(self, x, debug=False):
        x_3d = x.reshape(-1, 28, 28)
        logging.info('Compute grad wrt. X shape %s' % (x_3d.shape,))

        with self.get_session() as sess:
            rx = np.zeros((x_3d.shape[0], self.architecture.recur))

            max_y_pred = tf.reduce_max(self.dag.y_pred, axis=1)

            grad = tf.gradients(max_y_pred, self.dag.x)

            pred, grad_res = sess.run([self.dag.y_pred, grad],
                                      feed_dict={self.dag.x: x_3d, self.dag.rx: rx, self.dag.keep_prob: 1})

            grad_res = grad_res[0]

            logging.info('Grad result in shape %s' % (grad_res.shape,))

            return np.argmax(pred, axis=1), grad_res

    def rel_sensitivity(self, x, debug=False):
        pred, grad = self.compute_grad_wrt_x(x, debug)
        return pred, np.power(grad, 2)

    def rel_simple_taylor(self, x, debug=False):
        x_3d = x.reshape(-1, 28, 28)

        pred, grad = self.compute_grad_wrt_x(x_3d, debug)

        return pred, grad * x_3d

    def rel_lrp_deep_taylor(self, x, debug=False):
        return self.lrp(x, factor=1, debug=debug)

    def _get_relevance(self, x_3d):
        rx = np.zeros((x_3d.shape[0], self.architecture.recur))

        pred = sess.run(self.dag.y_pred, feed_dict={self.dag.x: x_3d, self.dag.rx: rx, self.dag.keep_prob: 1})
        mark = np.zeros(pred.shape)
        mark[range(pred.shape[0]), np.argmax(pred, axis=1)] = 1

        relevance = pred * mark

        return relevance

    def get_weight_bias_at_layers(self, layers=None):
        if layers is None:
            layers = sorted(self.dag.layers.keys())

        weights = []
        biases = []

        total_layers = len(layers)

        for k in layers:
            layer = self.dag.layers[k]
            weights.append(layer.W)
            biases.append(layer.b)

        with self.get_session() as sess:
            res = sess.run(weights + biases)

            weights, biases = res[:total_layers], res[total_layers:]

        return dict(zip(layers, weights)), dict(zip(layers, biases))
