import tensorflow as tf
import logging
import numpy as np
from utils import logging as lg

from utils import experiment_artifact, data_provider
from model.components import layer as ComponentLayer

from model import provider as model_provider

lg.set_logging()


class BaseDag:
    def __init__(self, architecture, dims, max_seq_length, optimizer, no_classes):
        tf.reset_default_graph()

        self.rx = tf.placeholder(tf.float32, shape=(None, architecture.recur), name='recurrent_input')
        self.x = tf.placeholder(tf.float32, shape=(None, dims, max_seq_length), name='input')
        self.x_with_channels = tf.expand_dims(self.x, -1)

        self.y_target = tf.placeholder(tf.float32, [None, no_classes], name='output_target')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.regularizer = tf.placeholder(tf.float32, name='regularizer')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # loss and optimizer
        self.loss_op = None
        self.train_op = None
        self.init_op = None
        self.accuracy = None

        self.layers = []

        self.optimizer = optimizer

        # lrp variables
        self.y_pred_reduced_1d = None
        self.total_relevance = None

    def setup_loss_and_opt(self):
        reg_term = tf.constant(0.0)

        for k, v in self.layers.items():
            if hasattr(v, 'W'):
                reg_term = reg_term + tf.reduce_sum(tf.pow(v.W, 2))

        self.loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_target) +
            self.regularizer * reg_term
        )

        optimizer = getattr(tf.train, self.optimizer)
        self.train_op = optimizer(learning_rate=self.lr).minimize(self.loss_op)
        self.init_op = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(self.y_target, 1), tf.argmax(self.y_pred, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', self.loss_op)
        tf.summary.scalar('accuracy', self.accuracy)

        self.summary = tf.summary.merge_all()

    def setup_variables_for_lrp(self):

        self.y_pred_reduced_1d = tf.reduce_max(self.y_pred, axis=1)
        mark = tf.cast(tf.equal(self.y_pred,
                                tf.reshape( self.y_pred_reduced_1d, (-1, 1))), tf.float32)

        self.total_relevance = mark*self.y_pred

    def no_variables(self):
        no_variables = 0
        for k, ly in self.layers.items():
            no_variables = no_variables + ly.get_no_variables()
        return no_variables


class BaseNetwork:
    def __init__(self, artifact: experiment_artifact.Artifact):

        self.experiment_artifact = artifact
        self._ = artifact

        self.data_no_rows, self.data_no_cols = self._.dims, self._.max_seq_length

        self.dag = None
        tf.reset_default_graph()

    def get_session(self):

        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, '%s/model.ckpt' % self._.path)

        return sess

    def compute_grad_wrt_x(self, x, debug=False):
        x_3d = x.reshape(-1, self.data_no_rows, self.data_no_cols)
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
        x_3d = x.reshape(-1, self.data_no_rows, self.data_no_cols)

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

        layers = []
        for k, v in self.dag.layers.items():
            if type(v) != ComponentLayer.PoolingLayer:
                layers.append(k)

        layers = sorted(layers)

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

    def _build_heatmap(self, sess, x, rr_of_pixels, debug=False):

        total_relevance_reduced = tf.reduce_sum(self.dag.total_relevance, axis=1)

        rx = np.zeros((x.shape[0], self.architecture.recur))
        pred, total_relevance, rr_of_pixels = sess.run(
            [self.dag.y_pred_reduced_1d, total_relevance_reduced, rr_of_pixels],
            feed_dict={self.dag.x: x, self.dag.rx: rx, self.dag.keep_prob: 1})

        relevance_heatmap = np.zeros(x.shape)
        for i in range(0, relevance_heatmap.shape[2], self._.column_at_a_time):
            t_idx = int(i / self._.column_at_a_time)
            relevance_heatmap[:, :, i:(i + self._.column_at_a_time)] = rr_of_pixels[t_idx] \
                .reshape(relevance_heatmap.shape[0], relevance_heatmap.shape[1], -1)

        if debug:

            logging.debug('Prediction before softmax')
            logging.debug(pred)
            logging.debug('Relevance')
            logging.debug(total_relevance)

            total_relevance_pixels = np.sum(relevance_heatmap, axis=(1, 2))
            np.testing.assert_allclose(total_relevance_pixels, total_relevance,
                                       rtol=1e-6, atol=0,
                                       err_msg='Conservation property isn`t hold\n'
                                               ': Sum of relevance from pixels is not equal to output relevance.')
        return pred, relevance_heatmap

    def formal_name(self):
        return '%s-%d' % (model_provider.network_nickname(self._.architecture_name), self._.seq_length)
