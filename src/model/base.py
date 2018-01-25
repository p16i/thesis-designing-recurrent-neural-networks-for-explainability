import tensorflow as tf
import logging
import numpy as np
from utils import logging as lg

from utils import experiment_artifact
from model.components import layer as ComponentLayer

lg.set_logging()


#### Acknowledgement to Chris Olah ####
@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return gate_y * gate_g * grad


class BaseDag:
    def __init__(self, architecture, dims, max_seq_length, optimizer, no_classes):

        self.no_classes = no_classes

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

        self.y_pred_y_target =  self.y_pred * self.y_target

    def setup_variables_for_lrp(self):

        self.total_relevance = tf.nn.relu(self.y_pred) * self.y_target

        # self.total_relevance = tf.max(tf.nn.relu(self.y_pred), axis=1)

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

    def compute_grad_wrt_x(self, x, y, debug=False):
        x_3d = x.reshape(-1, self.data_no_rows, self.data_no_cols)
        logging.info('Compute grad wrt. X shape %s' % (x_3d.shape,))

        with self.get_session() as sess:
            rx = np.zeros((x_3d.shape[0], self.architecture.recur))

            relevance = self.dag.y_pred_y_target

            grad = tf.gradients(relevance, self.dag.x)

            pred, grad_res = sess.run([self.dag.y_pred, grad],
                                      feed_dict={self.dag.x: x_3d, self.dag.y_target: y,
                                                 self.dag.rx: rx, self.dag.keep_prob: 1})

            grad_res = grad_res[0]

            logging.info('Grad result in shape %s' % (grad_res.shape,))

            return np.argmax(pred, axis=1), grad_res

    def rel_sensitivity(self, x, y, debug=False):
        logging.info('Explaining with sensitivity')
        pred, grad = self.compute_grad_wrt_x(x, y, debug)
        return pred, np.power(grad, 2)

    def rel_simple_taylor(self, x, y, debug=False):
        logging.info('Explaining with simple taylor')
        x_3d = x.reshape(-1, self.data_no_rows, self.data_no_cols)
        pred, grad = self.compute_grad_wrt_x(x_3d, y, debug)

        return pred, grad * x_3d

    def rel_guided_backprop(self, x, y, debug=False):
        logging.info('Explaining with guided_backprop')
        x_3d = x.reshape(-1, self.data_no_rows, self.data_no_cols)

        tf.reset_default_graph()
        with tf.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}):
            dag = self.create_graph()

            saver = tf.train.Saver()
            sess = tf.Session()

            saver.restore(sess, '%s/model.ckpt' % self._.path)
            with sess:
                rx = np.zeros((x_3d.shape[0], self.architecture.recur))
                relevance = dag.y_pred_y_target

                grad = tf.gradients(relevance, dag.x)
                pred, grad_res = sess.run([dag.y_pred, grad],
                                          feed_dict={dag.x: x_3d, dag.y_target: y, dag.rx: rx, dag.keep_prob: 1})
                grad_res = grad_res[0]
        tf.reset_default_graph()
        self.dag = self.create_graph()
        return np.argmax(pred, axis=1), np.power(grad_res, 2)

    # def rel_integrated_grad(self, x, y, debug=False, m=50):
    #
    #     x_0 = np.zeros(x.shape)
    #     x_diff = x - x_0
    #     acc_grad = 0
    #     for k in np.linspace(0, 1, m):
    #         x_k = x_0 + k * x_diff
    #         pred, grad = self.compute_grad_wrt_x(x_k, debug=debug)
    #         acc_grad = acc_grad + grad
    #
    #     int_grad = acc_grad * x_diff
    #
    #     # print('approximation error %f' % (np.mean(np.sum(int_grad, axis=1) - x_diff, axis=1))))
    #
    #     return pred, int_grad

    def rel_lrp_deep_taylor(self, x, y, debug=False):
        logging.info('Explaining with deep_taylor')
        return self.lrp(x, y, alpha=1, beta=0.0, debug=debug)

    def rel_lrp_alpha3_beta2(self, x, y, debug=False):
        logging.info('Explaining with alpha=3, beta=2')
        return self.lrp(x, y, alpha=3, beta=2, debug=debug)

    def rel_lrp_alpha2_beta1(self, x, y, debug=False):
        logging.info('Explaining with alpha2, beta=1')
        return self.lrp(x, y, alpha=2.0, beta=1, debug=debug)

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

    def _build_heatmap(self, sess, x, y, rr_of_pixels, debug=False):

        total_relevance_reduced = tf.reduce_sum(self.dag.total_relevance, axis=1)

        rx = np.zeros((x.shape[0], self.architecture.recur))
        pred, total_relevance, rr_of_pixels = sess.run(
            [self.dag.y_pred, total_relevance_reduced, rr_of_pixels],
            feed_dict={self.dag.x: x, self.dag.y_target: y, self.dag.rx: rx, self.dag.keep_prob: 1})

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
        return np.argmax(pred, axis=1), relevance_heatmap

    def formal_name(self):
        return '%s-%d' % (BaseNetwork.network_nickname(self._.architecture_name), self._.seq_length)

    @staticmethod
    def network_nickname(t):
        if t == 's2_network':
            return 'Shallow'
        elif t == 's3_network':
            return 'Deep'
        elif t == 'deep_4l_network':
            return 'DeepV2'
        elif t == 'convdeep_4l_network':
            return 'ConvDeep'
        else:
            return t

    def create_graph(self):
        dag_class = self.dag.__class__
        return dag_class(self._.column_at_a_time, self.data_no_rows, self.data_no_cols, self.architecture,
                         self._.optimizer, self.dag.no_classes)
