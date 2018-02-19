import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import logging as lg
from model import provider
import numpy as np
import config
import pandas as pd

import seaborn as sns

lg.set_logging()

# relative path from notebook dir
FIGURE_PATH = '../figures/nb_figures'


def setup():
    logging.debug('Setup plot parameters')
    plt.show_and_save = show_and_save


def show_and_save(title=""):
    if title:
        path = '%s/%s' % (FIGURE_PATH, title)
        logging.debug('save fig to %s' % path)
        plt.savefig(path)
    plt.show()


def show_model_accuracy(dataset, models=['s2', 's3', 'deep_4l', 'convdeep_4l'], seqs=[1, 4, 7]):
    results = []
    print("%s accuracy" % dataset)
    model_names = list(map(lambda m: config.architecture_name(m), models))
    for seq in seqs:
        d = dict(seq=seq, **dict(zip(model_names, [-1] * len(model_names))))
        for model in models:
            try:
                model_obj = provider.load('.%s' % provider._model_path(model, dataset, seq))
                d[config.architecture_name(model)] = model_obj._.accuracy
            except:
                pass

        results.append(d)
    return pd.DataFrame(results)[['seq'] + model_names]


def plot_relevance_dist_at_between_timestep(datasets=['mnist-3-digits'], between=(4, 8)):
    results = []
    for dataset in datasets:
        file = "../stats/rel-dist-%s.pkl" % (dataset)
        print('getting data from %s' % file)
        results = results + pickle.load(open(file, "rb"))

    df = pd.DataFrame(results)

    def compute_dist(row):
        return np.sum(row['rel_dist'][between[0]:between[1]])

    df['architecture_idx'] = df['architecture'].apply(architecture_idx)

    col_name = '%% relevance between t=(%d,%d)' % between
    df[col_name] = df.apply(compute_dist, axis=1)

    g = sns.factorplot(x="architecture_idx", y=col_name, col='dataset', hue="method",
                       data=df, size=5, markers=['.', '.', 's', 'o', 'o', '^'],
                       linestyles=[':', ':', ':', '-', '-', '-'])

    g.set_xticklabels(['Shallow', 'Deep', 'DeepV2', 'ConvDeep'], rotation=30)
    g.set(xlabel='')


def plot_relevance_methods(model_path, dataset_loader,
                           methods=['sensitivity', 'simple_taylor', 'guided_backprop',
                                    'lrp_alpha2_beta1', 'lrp_alpha3_beta2', 'lrp_deep_taylor'],
                           skip_data=False, overlay=False, data=None, verbose=False, only_positive_rel=False
                           ):

    model_obj = provider.load(model_path)

    dataset = dataset_loader.load(model_obj._.dataset)

    if data is None:
        data, label = dataset.get_samples_for_vis()
    else:
        data, label = data
    total = data.shape[0]

    if not skip_data:
        methods = [None] + methods
    total_methods = len(methods)

    fig = plt.figure(figsize=(20, 2 * total_methods))

    outer = gridspec.GridSpec(total_methods, 1)

    pred_heatmaps = dict()
    actual_methods = list(filter(lambda x: x, methods))
    for method in actual_methods:
        pred_heatmaps[method] = getattr(model_obj, 'rel_%s' % method)(data, label, debug=verbose)

    for i in range(total_methods):
        inner = gridspec.GridSpecFromSubplotSpec(1, total,
                                                 subplot_spec=outer[i], wspace=0.1, hspace=0.05)

        method = methods[i]

        for j in range(total):
            ax = plt.Subplot(fig, inner[j])
            if method is not None:
                heatmap = pred_heatmaps[method][1][j, :, :]
                if verbose:
                    print('Relevance score for %d  : %.4f' % (j, np.sum(heatmap)))

                heatmap = heatmap / (np.abs(heatmap).max() + 1e-10)

                if only_positive_rel:
                    heatmap = heatmap * (heatmap > 0)

                heatmap = make_rgb_heatmap(heatmap)
                cmap = None
                if overlay:
                    hmin = np.min(heatmap)
                    hmax = np.max(heatmap)
                    heatmap = (heatmap - hmin) / (hmax - hmin) * data[j, :]
            else:
                heatmap = data[j, :]
                pred_idx = pred_heatmaps[actual_methods[0]][0][j]
                ax.set_title('Pred\n%s(%d)' % (dataset.get_text_label(pred_idx), pred_idx))
                cmap = 'gist_gray'

            ax.imshow(heatmap, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])

            fig.add_subplot(ax)

        ax_outer = plt.Subplot(fig, outer[i])
        if method is not None:
            title = method
        else:
            title = 'data'

        ax_outer.set_title('\n%s\n' % title)

        ax_outer._frameon = False
        ax_outer.xaxis.set_visible(False)
        ax_outer.yaxis.set_visible(False)
        fig.add_subplot(ax_outer)

    plt.suptitle(
        'Heatmaps from different explaination methods\n%s:%s\n%s (no. variables %d ) \n(opt %s, acc %.4f, keep_prob %.2f)' %
        (
            config.architecture_name(model_obj._.architecture_name),
            model_obj._.architecture,
            model_obj._.experiment_name,
            model_obj.dag.no_variables(),
            model_obj._.optimizer,
            model_obj._.accuracy,
            model_obj._.keep_prob
        ), y=1.1)

    plt.tight_layout()

    plt.show()


def architecture_idx(a):
    if a == 's2':
        return 1
    elif a == 's3':
        return 2
    elif a == 'deep_4l':
        return 3
    elif a == 'convdeep_4l':
        return 4

# taken from http://heatmapping.org/tutorial/utils.py.txt
def make_rgb_heatmap(x):
	x = x[...,np.newaxis]

	# positive relevance
	hrp = 0.9 - np.clip(x-0.3,0,0.7)/0.7*0.5
	hgp = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4
	hbp = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4

	# negative relevance
	hrn = 0.9 - np.clip(-x-0.0,0,0.3)/0.3*0.5 - np.clip(-x-0.3,0,0.7)/0.7*0.4
	hgn = 0.9 - np.clip(-x-0.0,0,0.3)/0.3*0.5 - np.clip(-x-0.3,0,0.7)/0.7*0.4
	hbn = 0.9 - np.clip(-x-0.3,0,0.7)/0.7*0.5

	r = hrp*(x>=0)+hrn*(x<0)
	g = hgp*(x>=0)+hgn*(x<0)
	b = hbp*(x>=0)+hbn*(x<0)

	return np.concatenate([r,g,b],axis=-1)

def norm_and_make_rgb_heatmap(x):

    x = x / (np.abs(x).max() + 1e-10)

    return make_rgb_heatmap(x)
