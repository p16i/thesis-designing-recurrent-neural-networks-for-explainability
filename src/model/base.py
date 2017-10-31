import tensorflow as tf


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

        self.optimizer = optimizer

    def setup_loss_and_opt(self):
        self.loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_target))

        optimizer = getattr(tf.train, self.optimizer)
        self.train_op = optimizer(learning_rate=self.lr).minimize(self.loss_op)
        self.init_op = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(self.y_target, 1), tf.argmax(self.y_pred, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


class BaseNetwork:
    def __init__(self, artifact):

        self.experiment_artifact = artifact
        self._ = artifact
        tf.reset_default_graph()

    def get_session(self):

        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, '%s/model.ckpt' % self._.path)

        return sess

