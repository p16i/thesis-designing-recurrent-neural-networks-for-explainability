import tensorflow as tf
import logging
import numpy as np
from utils import logging as lg

from utils import experiment_artifact
from model.components import layer as ComponentLayer

lg.set_logging()

TEST_RELEVANCE_THRESHOLD = 1e-2


COMPUTE_BATCH_SIZE = 200

#### Acknowledgement to Chris Olah ####
@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return gate_y * gate_g * grad


class BaseDag:
    def __init__(self, architecture, dims, max_seq_length, optimizer, no_classes):

        self.no_classes = no_classes

        self.x = tf.placeholder(tf.float32, shape=(None, dims, max_seq_length), name='input')
        self.x_with_channels = tf.expand_dims(self.x, -1)
        self.rx = tf.zeros((tf.shape(self.x)[0], architecture.recur))

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

        size = (tf.shape(self.y_pred)[0], 1)
        pred_pseudo_class = tf.zeros(size)
        target_pseudo_class = tf.zeros(size)

        y_pred_with_psuedo_class = tf.concat([self.y_pred, pred_pseudo_class], axis=1)
        y_target_with_psuedo_class = tf.concat([self.y_target, target_pseudo_class], axis=1)

        self.loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_with_psuedo_class,
                                                    labels=y_target_with_psuedo_class)
            + self.regularizer * reg_term
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
            logging.info('layer %s | # variables %d' % (k, ly.get_no_variables()))
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

        pred = np.zeros(y.shape)
        grad_res = np.zeros(x.shape)

        with self.get_session() as sess:

            relevance = self.dag.y_pred_y_target

            grad = tf.gradients(relevance, self.dag.x)

            for i in range(0, x.shape[0], COMPUTE_BATCH_SIZE):

                st = i
                sp = np.min([i+COMPUTE_BATCH_SIZE, x.shape[0]])
                logging.info('data indices [%d, %d)' % (st, sp))

                pred_cur, grad_res_cur = sess.run([self.dag.y_pred, grad],
                                                  feed_dict={self.dag.x: x_3d[st:sp, :, :],
                                                             self.dag.y_target: y[st:sp, :],
                                                             self.dag.keep_prob: 1})

                pred[st:sp, :] = pred_cur
                grad_res[st:sp, :, :] = grad_res_cur[0]

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

        pred = np.zeros(y.shape)
        grad_res = np.zeros(x.shape)
        with tf.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}):
            dag = self.create_graph()

            saver = tf.train.Saver()
            sess = tf.Session()

            saver.restore(sess, '%s/model.ckpt' % self._.path)
            with sess:
                relevance = dag.y_pred_y_target

                grad = tf.gradients(relevance, dag.x)

                for i in range(0, x.shape[0], COMPUTE_BATCH_SIZE):
                    st = i
                    sp = np.min([i+COMPUTE_BATCH_SIZE, x.shape[0]])

                    logging.info('data indices [%d, %d)' % (st, sp))

                    pred_cur, grad_res_cur = sess.run([dag.y_pred, grad],
                                                      feed_dict={dag.x: x_3d[st:sp, :, :],
                                                                 dag.y_target: y[st:sp, :],
                                                                 dag.keep_prob: 1})

                    pred[st:sp, :] = pred_cur
                    grad_res[st:sp, :, :] = grad_res_cur[0]

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

    def rel_lrp_alpha1_5_beta_5(self, x, y, debug=False):
        logging.info('Explaining with alpha=1.5, beta=0.5')
        return self.lrp(x, y, alpha=1.5, beta=0.5, debug=debug)

    def rel_lrp_alpha1_2_beta_2(self, x, y, debug=False):
        logging.info('Explaining with alpha1.2, beta=0.2')
        return self.lrp(x, y, alpha=1.2, beta=0.2, debug=debug)


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

        pred = np.zeros(y.shape)
        total_relevance = np.zeros(x.shape[0])
        rr_of_pixels_store = np.zeros((x.shape[0], len(rr_of_pixels), self._.column_at_a_time*self._.dims))

        for i in range(0, x.shape[0], COMPUTE_BATCH_SIZE):

            st = i
            sp = np.min([i+COMPUTE_BATCH_SIZE, x.shape[0]])

            logging.info('data indices [%d, %d)' % (st, sp))

            pred_cur, total_relevance_cur, rr_of_pixels_cur = sess.run(
                [self.dag.y_pred, total_relevance_reduced, rr_of_pixels],
                feed_dict={self.dag.x: x[st:sp, :, :], self.dag.y_target: y[st:sp, :],
                           self.dag.keep_prob: 1})

            pred[st:sp, :] = pred_cur
            total_relevance[st:sp] = total_relevance_cur
            rr_of_pixels_store[st:sp, :, :] = np.array(rr_of_pixels_cur).transpose([1, 0, 2])

        relevance_heatmap = np.zeros(x.shape)
        for i in range(0, relevance_heatmap.shape[2], self._.column_at_a_time):
            t_idx = int(i / self._.column_at_a_time)
            relevance_heatmap[:, :, i:(i + self._.column_at_a_time)] = rr_of_pixels_store[:, t_idx, :] \
                .reshape(relevance_heatmap.shape[0], relevance_heatmap.shape[1], -1)

        if debug:
            print('debug')

            logging.debug('Prediction before softmax')
            logging.debug(pred)
            logging.debug('Relevance')
            logging.debug(total_relevance)

            total_relevance_pixels = np.sum(relevance_heatmap, axis=(1, 2))

            diff_greater_than_threshold = (np.abs(total_relevance_pixels - total_relevance) > TEST_RELEVANCE_THRESHOLD)
            no_diff_greater_than_threshold = np.sum(diff_greater_than_threshold)

            for i in range(total_relevance_pixels.shape[0]):
                rp = total_relevance_pixels[i]
                re = total_relevance[i]
                print('%d: out: %f \t\t | exp: %f \t\t diff %f(%s)'
                      % (i, rp, re, rp-re, diff_greater_than_threshold[i]))

            print('there are %d diffs greather than threshold %f' % (no_diff_greater_than_threshold, TEST_RELEVANCE_THRESHOLD))
            assert no_diff_greater_than_threshold == 0,\
                'Conservation property isn`t hold\n : Sum of relevance from pixels is not equal to output relevance.'
        return np.argmax(pred, axis=1), relevance_heatmap

    def create_graph(self):
        dag_class = self.dag.__class__
        return dag_class(self._.column_at_a_time, self.data_no_rows, self.data_no_cols, self.architecture,
                         self._.optimizer, self.dag.no_classes)
