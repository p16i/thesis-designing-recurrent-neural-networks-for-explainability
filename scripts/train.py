
import logging
import fire

import numpy as np
import tensorflow as tf

from model import s2_network, s3_network, deep_4l_network, convdeep_4l_network
from utils import logging as lg
from utils import data_provider, experiment_artifact, network_architecture

lg.set_logging()

NETWORKS = {
    's2_network': s2_network,
    's3_network': s3_network,
    'deep_4l_network': deep_4l_network,
    'convdeep_4l_network': convdeep_4l_network
}


def train(network, seq_length=1, epoch=1, lr=0.01, batch=100, keep_prob=0.5, architecture_str='hidden:_|out:_|--recur:_',
          verbose=False, output_dir='./experiment-result', optimizer='AdamOptimizer', dataset='mnist', regularizer=0.0
          ):

    experiment_name = experiment_artifact.get_experiment_name('%s-%s-seq-%d--' % (network, dataset, seq_length))

    logging.debug('Train %s' % network)
    logging.debug('Experiment name : %s' % experiment_name)
    data = data_provider.get_data(dataset)

    # no.rows and cols
    dims, max_seq_length = data.train2d.x.shape[1:]
    architecture = NETWORKS[network].Architecture(**network_architecture.parse(architecture_str))
    logging.debug('Network architecture')
    logging.debug(architecture)

    no_input_cols = max_seq_length // seq_length
    logging.debug('Training %d columns at a time' % no_input_cols)
    logging.debug('Optimizer %s' % optimizer)

    dag = NETWORKS[network].Dag(no_input_cols, dims, max_seq_length, architecture, optimizer)

    with tf.Session() as sess:
        sess.run(dag.init_op)
        step = 1
        for i in range(epoch):
            logging.debug('epoch %d' % (i + 1))
            for bx, by in data.train2d.get_batch(no_batch=batch):

                rx0 = np.zeros((batch, architecture.recur))
                sess.run(dag.train_op,
                         feed_dict={dag.x: bx, dag.y_target: by, dag.rx: rx0, dag.lr: lr,
                                    dag.keep_prob: keep_prob, dag.regularizer: regularizer})

                if (step % 1000 == 0 or step < 10) and verbose:
                    rx0 = np.zeros((len(by), architecture.recur))
                    acc, loss = sess.run([dag.accuracy, dag.loss_op],
                                         feed_dict={dag.x: bx, dag.y_target: by, dag.rx: rx0, dag.keep_prob: 1,
                                                    dag.regularizer: regularizer
                                                    })

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
            architecture_name=network,
            dims=dims,
            max_seq_length=max_seq_length,
            keep_prob=keep_prob,
            optimizer=optimizer,
            val_accuracy=val_acc,
            dataset=dataset,
            regularizer=regularizer
        )

        logging.debug('\n%s\n', lg.tabularize_params(res))

        output_dir = '%s/%s' % (output_dir, experiment_name)

        return experiment_artifact.save_artifact(sess, res, output_dir=output_dir)


if __name__ == '__main__':
    fire.Fire(train)
