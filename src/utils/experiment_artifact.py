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
                       'val_accuracy', 'keep_prob', 'optimizer', 'dataset', 'regularizer']
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

    if not os.path.isdir(output_dir):
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

    if 'keep_prob' not in res.keys():
        res['keep_prob'] = 1

    if 'optimizer' not in res.keys():
        res['optimizer'] = 'AdamOptimizer'

    if 'val_accuracy' not in res.keys():
        res['val_accuracy'] = -1

    if 'dataset' not in res.keys():
        res['dataset'] = 'mnist'

    if 'regularizer' not in res.keys():
        res['regularizer'] = 0.0

    return Artifact(**res)


def get_experiment_name(prefix='rnn'):
    timestamp = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

    return '%s-%s' % (prefix, timestamp)
