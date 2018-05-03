
import logging
import fire

import numpy as np
import tensorflow as tf

from model import provider
from utils import logging as lg
from utils import data_provider, experiment_artifact, network_architecture
import compute_stats

lg.set_logging()


def train(architecture='<network>::<architecture_str>', seq_length=1, epoch=1, lr=0.01, batch=100, keep_prob=0.5,
          verbose=False, output_dir='./experiment-result', optimizer='AdamOptimizer', dataset='mnist', regularizer=0.0,
          cv_folds=0, should_compute_stat=True
          ):

    network, architecture_str = architecture.split('::')
    logging.debug('Train %s' % network)

    experiment_name_base = experiment_artifact.get_experiment_name('%s-%s-seq-%d--' % (network, dataset, seq_length))

    # no.rows and cols
    architecture_class = provider.get_architecture_class(network)
    architecture = architecture_class.Architecture(**network_architecture.parse(architecture_str))
    logging.debug('Network architecture')
    logging.debug(architecture)

    data = data_provider.DatasetLoader(data_dir='./data').load(dataset)
    dims, max_seq_length = data.train2d.x.shape[1:]

    no_input_cols = max_seq_length // seq_length
    logging.debug('Training %d columns at a time' % no_input_cols)
    logging.debug('Optimizer %s' % optimizer)

    if cv_folds >= 2:
        # build cv data
        logging.info('Building %d-fold data' % cv_folds)
        cv_datasets = data_provider.build_cvdataset(data, k=cv_folds)
    else:
        logging.info('Using original training & testing set')
        cv_datasets = [(data.train2d, data.val2d, data.test2d)]

    for (fold, (dtrain, dval, dtest)) in enumerate(cv_datasets):
        tf.reset_default_graph()

        if cv_folds >= 2:
            experiment_name = '%s--fold-%d' % (experiment_name_base, fold)
        else:
            experiment_name = experiment_name_base

        logging.debug('Experiment name : %s' % experiment_name)

        output_dir_run = '%s/%s' % (output_dir, experiment_name)

        dag = architecture_class.Dag(no_input_cols, dims, max_seq_length, architecture, optimizer, data.no_classes)
        print('no. variables %d' % dag.no_variables())

        train_writer = tf.summary.FileWriter(output_dir_run + '/boards/train')
        val_writer = tf.summary.FileWriter(output_dir_run + '/boards/validate')

        print('-'*100)
        print('Tensorboard : tensorboard  --logdir %s' % output_dir_run)
        print('-'*100)

        with tf.Session() as sess:

            sess.run(dag.init_op)
            step = 1
            for i in range(epoch):
                for bx, by in dtrain.get_batch(no_batch=batch):

                    sess.run(dag.train_op,
                             feed_dict={dag.x: bx, dag.y_target: by, dag.lr: lr,
                                        dag.keep_prob: keep_prob, dag.regularizer: regularizer})

                    if (step % 100 == 0 or step < 10) and verbose:
                        summary, acc, loss = sess.run([dag.summary, dag.accuracy, dag.loss_op],
                                                      feed_dict={
                                                          dag.x: bx, dag.y_target: by,
                                                          dag.keep_prob: 1, dag.regularizer: regularizer
                                                      }
                                                      )

                        train_writer.add_summary(summary, step)
                        rx0 = np.zeros((dval.y.shape[0], architecture.recur))

                        summary, acc_val = sess.run([dag.summary, dag.accuracy],
                                                    feed_dict={dag.x: dval.x, dag.y_target: dval.y,
                                                                    dag.rx: rx0, dag.keep_prob: 1,
                                                               dag.regularizer: regularizer})
                        val_writer.add_summary(summary, step)
                        print('>>Fold-%d | Epoch %d | step %d : current train batch acc %f, loss %f | val acc %f'
                              % (fold, i, step, acc, loss, acc_val), end='\r', flush=True)

                    step = step + 1

            print('>>Fold-%d | Epoch %d | step %d : current train batch acc %f, loss %f | val acc %f'
                  % (fold, i, step, acc, loss, acc_val))
            # done training
            logging.debug('Calculating test accuracy')
            acc = float(sess.run(dag.accuracy,
                                 feed_dict={dag.x: dtest.x, dag.y_target: dtest.y, dag.keep_prob: 1}))

            val_acc = float(sess.run(dag.accuracy, feed_dict={dag.x: dval.x, dag.y_target: dval.y, dag.keep_prob: 1}))

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

            artifact = experiment_artifact.save_artifact(sess, res, output_dir=output_dir_run)

        if should_compute_stat:
            compute_stats.relevance_distribution(model_path=output_dir_run, data=dtest)

        logging.debug('\n%s\n', lg.tabularize_params(res))

    return artifact


if __name__ == '__main__':
    fire.Fire(train)
