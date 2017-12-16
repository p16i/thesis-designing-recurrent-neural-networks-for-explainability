import fire
import logging

import numpy as np

from model import provider
from utils import data_provider
from utils import logging as lg
lg.set_logging()


def train_more(model_path, epoch, lr, verbose=False):
    logging.info('Train %s with more %d epochs' % (model_path, epoch))

    model_obj = provider.load(model_path)

    mnist = data_provider.MNISTData()

    architecture = model_obj.architecture
    dag = model_obj.dag
    batch = model_obj._.batch
    lr = lr
    keep_prob = model_obj._.keep_prob

    with model_obj.get_session() as sess:
        step = 1
        for i in range(epoch):
            logging.debug('epoch %d' % (model_obj._.epoch + i + 1))
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

        logging.debug('Calculating test accuracy')
        rx0 = np.zeros((len(mnist.test2d.y), architecture.recur))
        acc = float(sess.run(dag.accuracy,
                             feed_dict={dag.x: mnist.test2d.x, dag.y_target: mnist.test2d.y,
                                        dag.rx: rx0, dag.keep_prob: 1}))

        rx0 = np.zeros((len(mnist.val2d.y), architecture.recur))
        val_acc = sess.run(dag.accuracy, feed_dict={dag.x: mnist.val2d.x, dag.y_target: mnist.val2d.y,
                                                    dag.rx: rx0, dag.keep_prob: 1})

        # TODO:
        # res = dict(
        #     experiment_name=experiment_name,
        #     seq_length=seq_length,
        #     epoch=epoch,
        #     column_at_a_time=no_input_cols,
        #     batch=batch,
        #     accuracy=acc,
        #     lr=lr,
        #     architecture=architecture_str,
        #     architecture_name='s3_network',
        #     dims=dims,
        #     max_seq_length=max_seq_length,
        #     keep_prob=keep_prob,
        #     optimizer=optimizer,
        #     val_accuracy=val_acc
        # )

        # logging.debug('\n%s\n', lg.tabularize_params(res))

    # load model

    #


if __name__ == '__main__':
    fire.Fire(train_more)
