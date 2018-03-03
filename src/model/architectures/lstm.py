from collections import namedtuple

import numpy as np
import tensorflow as tf

from model.architectures import base
from model.components.layer import Layer
from utils import experiment_artifact
from utils import logging as lg
from utils import network_architecture

lg.set_logging()


Architecture = namedtuple('Architecture', ['size', 'out', 'recur'])


def load(model_path):
    return Network.load(model_path)


# from https://github.com/ArrasL/LRP_for_LSTM/blob/master/code/LSTM/LRP_linear_layer.py
def lrp_linear(hin, w, b, hout, Rout, eps=0.0001, bias_factor=1.0, debug=False):
    """
    LRP for a linear layer with input dim D and output dim M.
    Args:
    - hin:            forward pass input, of shape (D,)
    - w:              connection weights, of shape (D, M)
    - b:              biases, of shape (M,)
    - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
    - Rout:           relevance at layer output, of shape (M,)
    - bias_nb_units:  number of lower-layer units onto which the bias/stabilizer contribution is redistributed
    - eps:            stabilizer (small positive number)
    - bias_factor:    for global relevance conservation set to 1.0, otherwise 0.0 to ignore bias redistribution
    Returns:
    - Rin:            relevance at layer input, of shape (D,)
    """
    sign_out = tf.where(hout >= 0, tf.ones(tf.shape(hout)), -tf.ones(tf.shape(hout)))  # shape (1, M)
    # print('------------')
    # print('singout_', sign_out.get_shape())


    Rj_normed = Rout / (hout + eps*sign_out)
    # print('Rj_normed', Rj_normed.get_shape())

    bias_nb_units = w.get_shape()[0].value

    Rj_bias_term = (eps*sign_out + bias_factor*b)/bias_nb_units*1.
    # Rj_bias_term = tf.ones(tf.shape(tf.transpose(w))) * tf.transpose(tf)
    # print('Rj_bias', Rj_bias_term.get_shape())


    ww = tf.matmul(Rj_normed, tf.transpose(w))
    # print('ww', ww.get_shape())

    bb = tf.matmul(Rj_bias_term * Rj_normed, tf.ones(tf.shape(tf.transpose(w))))
    # print('bb', bb.get_shape())

    # print('hin', hin.get_shape())

    message = hin * ww + bb
    # print('message', message.get_shape())
    # aw = hin * tf.multiply(w, tf.transpose(hin)) # 324x128
    # print('aw')
    # print(aw.get_shape())
    #
    # message = tf.matmul(Rj_normed, tf.transpose(aw + Rj_bias_term))
    # print('message')
    # print(message.get_shape())

    return message




    # # print('signout')
    # # print(sign_out.get_shape())
    # print('hin')
    # print(hin.get_shape())
    #
    # # print('hinw')
    # # hinw = tf.expand_dims(tf.multiply(hin, w), 0)
    # # print(hinw.get_shape())
    #
    # numer = tf.expand_dims(tf.multiply(w, tf.transpose(hin)),0) + (
    #     (bias_factor * b * 1. + eps * sign_out * 1.) * 1. / bias_nb_units)  # shape (D, M)
    # # numer =
    # # print('numer')
    # # print(numer.get_shape())
    #
    # denom = hout + (eps * sign_out * 1.)  # shape (1, M)
    # # print('denom')
    # # print(denom.get_shape())
    #
    # message = (numer / denom) * Rout  # shape (D, M)
    # # print('Rout')
    # # print(Rout.get_shape())
    #
    # print('message')
    # print(message.get_shape())
    # Rin = tf.reduce_sum(message, axis=2)
    # print('Rin')
    # print(Rin.get_shape())
    # #
    # # # Note: local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D
    # # #       global network relevance conservation if bias_factor==1.0 (can be used for sanity check)
    # # if debug:
    # #     print("local diff: ", Rout.sum() - Rin.sum())

    # return Rin


class Dag(base.BaseDag):
    def __init__(self, no_input_cols, dims, max_seq_length, architecture: Architecture, optimizer, no_classes):
        super(Dag, self).__init__(architecture, dims, max_seq_length, optimizer=optimizer, no_classes=no_classes)

        with tf.variable_scope('LSTM') as vs:
            ly_input_gate = Layer((no_input_cols * dims + architecture.size, architecture.size),
                                  default_biases=tf.Variable(tf.zeros([1, architecture.size])),
                                  name='input_gate')
            ly_forget_gate = Layer((no_input_cols * dims + architecture.size, architecture.size),
                                   default_biases=tf.Variable(tf.zeros([1, architecture.size])),
                                   name='forget_gate')
            ly_output_gate = Layer((no_input_cols * dims + architecture.size, architecture.size),
                                   default_biases=tf.Variable(tf.zeros([1, architecture.size])),
                                   name='output_gate')

            ly_new_cell_state = Layer((no_input_cols * dims + architecture.size, architecture.size),
                                      default_biases=tf.Variable(tf.zeros([1, architecture.size])),
                                      name='new_cell_state')

            ly_final_output = Layer((architecture.size, architecture.out),
                                    default_biases=tf.Variable(tf.zeros(([1, architecture.out]))),
                                    name='lstm_final_output')

            self.layers = {
                'input_gate': ly_input_gate,
                'forget_gate': ly_forget_gate,
                'output_gate': ly_output_gate,
                'final_output': ly_final_output,
                'new_cell_state': ly_new_cell_state
            }

            print('No. of variables %d' % self.no_variables())

            self.activation_labels = ['input_gate', 'forget_gate', 'output_gate', 'xh',
                                      'input_cell_state', 'new_cell_state', 'output']

            self.activations = namedtuple('Activations', self.activation_labels) \
                (**dict([(k, []) for k in self.activation_labels]))

            ct = tf.zeros([tf.shape(self.x)[0], architecture.size])
            ht = tf.zeros([tf.shape(self.x)[0], architecture.size])

            self.activations.new_cell_state.append(ct)

            # define  dag
            for i in range(0, max_seq_length, no_input_cols):
                xt = tf.reshape(self.x[:, :, i:i + no_input_cols], [-1, no_input_cols * dims])

                xh = tf.concat([xt, ht], axis=1)
                self.activations.xh.append(xh)

                ig = tf.sigmoid(tf.matmul(xh, ly_input_gate.W) + ly_input_gate.b)
                ig_do = tf.nn.dropout(ig, keep_prob=self.keep_prob)
                self.activations.input_gate.append(ig)

                fg = tf.sigmoid(tf.matmul(xh, ly_forget_gate.W) + ly_forget_gate.b)
                fg_do = tf.nn.dropout(fg, keep_prob=self.keep_prob)
                self.activations.forget_gate.append(fg)

                og = tf.sigmoid(tf.matmul(xh, ly_output_gate.W) + ly_output_gate.b)
                og_do = tf.nn.dropout(og, keep_prob=self.keep_prob)
                self.activations.output_gate.append(og)

                new_c = tf.tanh(tf.matmul(xh, ly_new_cell_state.W) + ly_new_cell_state.b)
                self.activations.input_cell_state.append(new_c)
                new_c_do = tf.nn.dropout(new_c, keep_prob=self.keep_prob)

                ct = ct*fg_do + new_c_do*ig_do
                ct_do = tf.nn.dropout(ct, keep_prob=self.keep_prob)
                self.activations.new_cell_state.append(ct)

                ht = og_do*tf.tanh(ct_do)
                self.activations.output.append(ht)

            ht_do = tf.nn.dropout(ht, keep_prob=self.keep_prob)
            self.y_pred = tf.matmul(ht_do, ly_final_output.W) + ly_final_output.b

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

        self.name = 'lstm'

    def rel_guided_backprop(self, x, y, debug=False):
        return np.zeros(y.shape[0]), np.zeros(x.shape)

    def rel_lrp_deep_taylor(self, x, y, debug=False):
        return self.rel_guided_backprop(x, y, debug)

    def lrp(self, x, y, alpha=1.0, beta=0.0, debug=False):

        with self.get_session() as sess:

            self.dag.setup_variables_for_lrp()

            rel_to_input = [None]*self._.seq_length

            rx = np.zeros((x.shape[0], self.architecture.recur))
            total_relevance_reduced = tf.reduce_sum(self.dag.total_relevance, axis=1)

            # NOTE: lwr start here
            rel_to_ht = self.dag.layers['final_output'].rel_z_plus_prop(
                self.dag.activations.output[-1],
                self.dag.total_relevance, beta=beta, alpha=alpha
            )


            rel_to_ct = rel_to_ht

            rel_to_g = lrp_linear(self.dag.activations.input_gate[-1] * self.dag.activations.input_cell_state[-1],
                                  tf.eye(self.architecture.size), tf.zeros(self.architecture.size),
                                  self.dag.activations.output[-1], rel_to_ct)


            rel_to_xh = lrp_linear(self.dag.activations.xh[-1],
                                   self.dag.layers['new_cell_state'].W, self.dag.layers['new_cell_state'].b,
                                   self.dag.activations.output[-1], rel_to_g)

            rel_to_input[-1] = rel_to_xh[:, :-self.architecture.size]
            rel_to_ht = rel_to_xh[:, -self.architecture.size:]

            rel_to_ct = lrp_linear(self.dag.activations.forget_gate[-1]*self.dag.activations.new_cell_state[-2],
                                   tf.eye(self.architecture.size), tf.zeros(self.architecture.size),
                                   self.dag.activations.new_cell_state[-1], rel_to_ct)

            rr_x, rr_ht, rr_ct, tt_rr = sess.run([rel_to_input[-1], rel_to_ht, rel_to_ct, total_relevance_reduced],
                                 feed_dict={self.dag.x: x, self.dag.y_target: y, self.dag.rx: rx, self.dag.keep_prob: 1})

            print(rr_x.shape)
            print('total relevance', tt_rr)
            print('rr_x relevance', np.sum(rr_x, axis=1))
            print('rr_ht relevance', np.sum(rr_ht, axis=1))
            print('rr_ct relevance', np.sum(rr_ct, axis=1))
            raise 'Force Exit'

            rel_to_ct = rel_to_ct + rel_to_ht



            for i in range(self._.seq_length - 1)[::-1]:
                rel_to_g = lrp_linear(self.dag.activations.input_gate[i] * self.dag.activations.input_cell_state[i],
                                      tf.eye(self.architecture.size), tf.zeros(self.architecture.size),
                                      self.dag.activations.output[i], rel_to_ct)

                rel_to_xh = lrp_linear(self.dag.activations.xh[i],
                                       self.dag.layers['new_cell_state'].W, self.dag.layers['new_cell_state'].b,
                                       self.dag.activations.output[i], rel_to_g)

                rel_to_input[i] = rel_to_xh[:, :-self.architecture.size]
                rel_to_ht = rel_to_xh[:, -self.architecture.size:]

                rel_to_ct = lrp_linear(self.dag.activations.forget_gate[i]*self.dag.activations.new_cell_state[i-1],
                                       tf.eye(self.architecture.size), tf.zeros(self.architecture.size),
                                       self.dag.activations.new_cell_state[i], rel_to_ct)

                rel_to_ct = rel_to_ct + rel_to_ht

            pred, heatmaps = self._build_heatmap(sess, x, y,
                                                 rr_of_pixels=rel_to_input, debug=debug)
        return pred, heatmaps

