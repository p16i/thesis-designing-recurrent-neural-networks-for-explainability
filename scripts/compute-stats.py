import logging

import fire
from utils import logging as lg
from utils import data_provider
from model import provider, heatmap_evaluation
import pandas as pd
import numpy as np
import pickle
import config


lg.set_logging()


SEQS = [1, 4, 7]
MODELS = ['s2', 's3', 'deep_4l', 'convdeep_4l']
METHODS = ['random', 'sensitivity', 'simple_taylor', 'guided_backprop',
           'lrp_alpha2_beta1', 'lrp_alpha3_beta2', 'lrp_deep_taylor']


DATASET = {
    'mnist': data_provider.MNISTData,
    'fashion-mnist': data_provider.FashionMNISTData,
    'ufi-cropped': data_provider.UFICroppedData
}


def no_flips(dataset, flip_function='minus_one'):
    raise SystemError('need to be verified and updated')
    logging.info('Computing no_flips for %s' % dataset)
    data = DATASET[dataset]()
    x = data.test2d.x
    y = data.test2d.y

    results = []
    for m in config.MODELS:
        for s in config.SEQS:
            model_obj = provider.load(_model_path(m, dataset, s))
            no_flip_random = heatmap_evaluation.count_flip(model_obj, x, y,
                                                           order='random',
                                                           method='sensitivity',
                                                           flip_function=flip_function
                                                           )
            results.append(dict(
                architecture=m,
                seq=s,
                method='random',
                dataset=dataset,
                no_flips=no_flip_random
            ))

            for e in METHODS:
                logging.info('>> %s-%d : %s' % (m, s, e))
                avg_flips = heatmap_evaluation.count_flip(model_obj, x, y, method=e, flip_function=flip_function)
                results.append(dict(
                    architecture=m,
                    seq=s,
                    method=e,
                    dataset=dataset,
                    no_flips=avg_flips
                ))

    pd.DataFrame(results).to_csv('./stats/no-flip-%s-using-%s-flip.csv' % (dataset, flip_function), index=False)


def aopc(dataset, flip_function='minus_one'):
    logging.info('Computing AOPC')
    data = DATASET[dataset]()
    x = data.test2d.x
    y = data.test2d.y

    results = []
    for m in config.MODELS:
        for s in config.SEQS:
            model_obj = provider.load(_model_path(m, dataset, s))

            for e in config.METHODS:
                logging.info('>> %s-%d : %s' % (m, s, e))
                order = 'morf'
                if e == 'random':
                    order = 'random'

                avg_relevance_at_k = heatmap_evaluation.aopc(model_obj, x, y, method=e,
                                                             order=order, flip_function=flip_function)
                results.append(dict(
                    architecture=m,
                    seq=s,
                    method=e,
                    dataset=dataset,
                    avg_relevance_at_k=avg_relevance_at_k
                ))

    output = './stats/aopc-%s-using-%s-flip.pkl' % (dataset, flip_function)
    with open(output, 'wb') as output:
        pickle.dump(results, output, -1)


def _model_path(network, dataset, seq):
    return './final-models/%s_network-%s-seq-%d' % (network, dataset, seq)


if __name__ == '__main__':
    fire.Fire()

