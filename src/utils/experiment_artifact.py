import logging
import glob
import yaml
import pandas as pd
import os
import tensorflow as tf
from datetime import datetime

from collections import namedtuple
from utils import logging as lg

lg.set_logging()

# TODO: Implement, object storage vs file-based

Artifact = namedtuple('ExperimentArtifact',
                      ['accuracy', 'architecture', 'batch', 'column_at_a_time', 'dims', 'epoch',
                       'experiment_name', 'lr', 'max_seq_length', 'seq_length', 'path', 'architecture_name',
                       'val_accuracy', 'keep_prob', 'optimizer']
                      )

def get_results(path):
    '''
    :param path:  path to a directory containing result files
    :return: pandas dataframe
    '''

    files = glob.glob(path + '/*/*.yaml')

    results = []
    for yf in files:
        with open(yf, 'r') as stream:
            try:
                res = yaml.load(stream)
                results.append(res)
            except yaml.YAMLError as exc:
                print(exc)

    return pd.DataFrame(results)


def save_artifact(tf_session, result, output_dir):

    os.makedirs(output_dir)

    result_path = '%s/result.yaml' % output_dir

    logging.debug('Saving result to %s' % result_path)
    with open(result_path, 'w') as outfile:
        yaml.dump(result, outfile, default_flow_style=False)

    model_path = '%s/model.ckpt' % output_dir
    logging.debug('Saving model to %s' % model_path)
    tf.train.Saver().save(tf_session, model_path)

    result['path'] = result_path

    return Artifact(**result)


def get_result(dir):
    result_path = '%s/result.yaml' % dir

    with open(result_path, 'r') as y:
        res = yaml.load(y)

    logging.debug('Getting result \n%s' % res)
    logging.debug(res)
    res['path'] = dir
    # res['parsed_architecture'] = network_architecture.parse_architecture(res['architecture'])

    if 'keep_prob' not in res.keys():
        res['keep_prob'] = 1

    if 'optimizer' not in res.keys():
        res['optimizer'] = 'AdamOptimizer'

    if 'val_accuracy' not in res.keys():
        res['val_accuracy'] = -1

    return Artifact(**res)


def get_experiment_name(prefix='rnn'):
    timestamp = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

    return '%s-%s' % (prefix, timestamp)


# class Artifact:
#     # | accuracy | 0.7764000296592712 |
#     # | architecture | hidden: 100 | out:10 - -recur: 100 |
#     # | architecture_name | type |
#     # | batch | 200 |
#     # | column_at_a_time | 4 |
#     # | dims | 28 |
#     # | epoch | 50 |
#     # | experiment_name | rnn - 2017 - 10 - 17 - -23 - 06 |
#     # | lr | 0.005 |
#     # | max_seq_length | 28 |
#     # | seq_length | 7 |
#     # +-------------------+--------------------
#     def __init__(self, accuracy, architecture, batch,
#                  column_at_a_time, dims, epoch, experiment_name, lr, max_seq_length, seq_length, path):
#

