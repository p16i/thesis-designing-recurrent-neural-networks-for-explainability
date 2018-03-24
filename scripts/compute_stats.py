import logging
import pickle

import fire
import numpy as np
import pandas as pd

import config
from model import provider, heatmap_evaluation
from utils import data_provider
from utils import logging as lg

lg.set_logging()

MODELS = ['s2', 's3', 'deep_4l', 'convdeep_4l']
METHODS = ['random', 'sensitivity', 'simple_taylor', 'guided_backprop',
           'lrp_alpha2_beta1', 'lrp_alpha3_beta2', 'lrp_deep_taylor']

DATASET = {
    'mnist': data_provider.MNISTData,
    'fashion-mnist': data_provider.FashionMNISTData,
    'ufi-cropped': data_provider.UFICroppedData,
}


def no_flips(dataset, flip_function='minus_one'):
    logging.info('Computing no_flips for %s' % dataset)
    data = DATASET[dataset]()
    x = data.test2d.x
    y = data.test2d.y

    results = []
    for m in MODELS:
        for s in SEQS:
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


def aopc(dataset, flip_function='minus_one', ref_model='conv-seq1', seqs=[1, 4, 7], dry_run=False, use_sample=False):
    logging.info('Computing AOPC')
    data = DATASET[dataset]()
    if use_sample:
        x, y = data.get_samples_for_vis(12)
    else:
        x = data.test2d.x
        y = data.test2d.y

    results = []
    for m in MODELS:
        for s in seqs:
            for e in METHODS:
                model_obj = provider.load(_model_path(m, dataset, s))

                logging.info('>> %s-%d : %s' % (m, s, e))
                order = 'morf'
                if e == 'random':
                    order = 'random'

                avg_relevance_at_k = heatmap_evaluation.aopc(model_obj, x, y, method=e,
                                                             order=order, flip_function=flip_function,
                                                             ref_model=ref_model)
                results.append(dict(
                    architecture=m,
                    seq=s,
                    method=e,
                    dataset=dataset,
                    avg_relevance_at_k=avg_relevance_at_k
                ))

    if not dry_run:
        output = './stats/auc-%s-morf-%s-model-using-%s-flip.pkl' % (dataset, ref_model, flip_function)
        with open(output, 'wb') as output:
            pickle.dump(results, output, -1)


def relevance_distribution(model_path, data=None, use_sample=False, dry_run=False):
    # TODO make this configurable
    original_data_width = 28

    logging.info('Computing relevance distribution for \n> %s' % model_path)
    model_obj = provider.load(model_path)

    if data is None:
        data_loader = data_provider.DatasetLoader(data_dir='./data')
        data = data_loader.load(model_obj._.dataset).test2d

    x, y, digit_mark = data.x, data.y, data.correct_digit_mark

    if use_sample:
        x, y, digit_mark = x[:10, :, :], y[:10, :], digit_mark[:10, :]

    methods = list(filter(lambda x: x != 'random', config.METHODS))

    total_digits = int(x.shape[2] / original_data_width)

    results = []

    steps_per_digit = int((original_data_width / x.shape[2]) * model_obj._.seq_length)
    logging.info('>>>>>>> steps per digit %d' % steps_per_digit)

    for e in methods:
        avg_rel, std_rel, raw_relevance = heatmap_evaluation.relevance_distributions(model_obj, x, y, method=e)

        total_relevance = np.sum(raw_relevance, axis=1).reshape(-1, 1)
        rel_dist = raw_relevance / (total_relevance + 1e-20)

        rel_dist_for_digits = np.zeros((raw_relevance.shape[0], total_digits))

        # aggregate to per digit-level
        for i in range(rel_dist_for_digits.shape[1]):
            st = i * steps_per_digit
            sp = (i + 1) * steps_per_digit
            print('digit %d : from t=[%d, %d)' % (i, st, sp))
            rel_dist_for_digits[:, i] = np.sum(rel_dist[:, st:sp], axis=1)

        relevance_of_correct_digits = rel_dist_for_digits * digit_mark

        adjusted_relevance_of_correct_digits = np.where(
            relevance_of_correct_digits <= config.MAX_RELEVANCE_PERCENTAGE_PER_SAMPLE, relevance_of_correct_digits,
            config.MAX_RELEVANCE_PERCENTAGE_PER_SAMPLE)
        print(adjusted_relevance_of_correct_digits.shape)
        adjusted_rel_dist_in_data_region = np.mean(np.sum(adjusted_relevance_of_correct_digits, axis=1), axis=0)
        print(adjusted_rel_dist_in_data_region)

        total_relevance_of_correct_digits = np.sum(relevance_of_correct_digits, axis=1)
        avg_std_total_dist = np.mean(np.std(relevance_of_correct_digits, axis=1))

        max_ss = np.max(relevance_of_correct_digits, axis=1).reshape(-1, 1)
        percentage_relevance_to_max = relevance_of_correct_digits / (max_ss + 1e-10)
        avg_percentage_relevance_to_max = np.mean(
            np.sum(percentage_relevance_to_max * (relevance_of_correct_digits < max_ss), axis=1))
        rel_dist_in_data_region = np.mean(total_relevance_of_correct_digits, axis=0)
        if use_sample:
            print('before multiplying mark %s' % e)
            print(rel_dist_for_digits)
            print('--------')
            print('after marking')
            print(relevance_of_correct_digits)
            print('Mean dist %f +/- std %f' % (rel_dist_in_data_region, avg_std_total_dist))
            print('=================')

        res = dict(
            architecture=model_obj._.architecture_name,
            seq=model_obj._.seq_length,
            dataset=model_obj._.dataset,
            method=e,
            rel_dist=avg_rel,
            std=std_rel,
            raw_relevance=raw_relevance,
            rel_dist_in_data_region=rel_dist_in_data_region,
            adjusted_rel_dist_in_data_region=adjusted_rel_dist_in_data_region,
            avg_std_total_dist=avg_std_total_dist,
            avg_percentage_relevance_to_max=avg_percentage_relevance_to_max
        )

        results.append(res)

    if not dry_run:
        with open('%s/rel-dist.pkl' % model_path, 'wb') as fp:
            pickle.dump(results, fp)

if __name__ == '__main__':
    fire.Fire()
