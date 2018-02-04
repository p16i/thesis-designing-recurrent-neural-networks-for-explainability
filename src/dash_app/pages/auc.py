from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc

import pandas as pd
import pickle

import os
import config
import subprocess
import numpy as np

cwd = os.getcwd()

AUC_FILEPATH = "%s/stats/auc-%s-morf-%s-model-using-%s-flip.pkl"

def norm_relevance(relevances):
    return (relevances - relevances[-1])/(relevances[0] - relevances[-1])


def build_graph(df, seq, model, methods=config.METHODS):
    print(seq, model)

    data = []
    min_rels = []
    for m in methods:
        ddf = df[(df.seq == seq) & (df.architecture == model) & (df.method == m)]
        min_rels.append(np.min(ddf.relevance))

    print('min for %s %d' % (model, seq))
    print(min_rels)

    min_rel = np.min(min_rels)
    for m in methods:
        ddf = df[(df.seq == seq) & (df.architecture == model) & (df.method == m)]
        relevances = norm_relevance(ddf.relevance.values)
        name = '%s(normed_auc=%.2f)' % (m, np.trapz(relevances))
        data.append(dict(x=ddf.k, y=relevances, name=name, mode='lines+markers',
                         marker={'symbol': config.METHOD_MARKERS[m]}))
    h = html.H4('%s' % (config.MODEL_NICKNAMES[model]))
    name = "auc-graph-%s-seq-%d" % (model, seq)
    g = dcc.Graph(id='%s-graph' % name, figure={
                  'data': data,
                  'layout': {'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}, 'showlegend': True,
                             'xaxis': dict(title='Flipping Step'), 'yaxis': dict(title='Relevance')}
              })

    return html.Div([h, g], id=name, style={'width': '25%', 'float': 'left', 'margin-bottom': '50px'})


def create_layout(ref_model, dataset, flip_function):
    print('creatinglayout')
    file = AUC_FILEPATH % (cwd, dataset, ref_model, flip_function)
    print('getting data from %s' % file)
    results = pickle.load(open(file, "rb"))

    total_k = len(results[0]['avg_relevance_at_k'])
    data = []
    for r in results:
        for i, rel in zip(range(total_k), r['avg_relevance_at_k']):
            d = dict(**r)
            d['k'] = i
            d['relevance'] = rel
            del d['avg_relevance_at_k']

            data.append(d)

    df = pd.DataFrame(data)

    graphs=[]
    for s in config.SEQS:
        h = html.H3('SEQ-LENGTH %d' % s)
        graphs.append(h)
        gs = [build_graph(df, s, m) for m in config.MODELS]

        graphs = graphs + gs

    return html.Div([
            _get_branch_div(),
            html.H1('Relevance Pixel-Flipping Curve %s' % (dataset.upper())),
            html.H4('using %s flip strategy and the relevances are computed from %s' % (flip_function, ref_model))
        ] + graphs)


def _get_branch_div():
    branch = subprocess.check_output('git branch  | grep "*"', shell=True).strip().decode("utf-8")
    return html.Div('branch : %s' % branch.split(' ')[1])


def create_summary_layout(ref_model, flip_function):

    def build_graph_dataset(dataset, seq):
        file = AUC_FILEPATH % (cwd, dataset, ref_model, flip_function)
        results = pickle.load(open(file, "rb"))

        data = []
        for r in results:
            r['auc'] = np.trapz(norm_relevance(r['avg_relevance_at_k']))
            data.append(r)

        df = pd.DataFrame(data)
        print(data[0])

        for method in config.METHODS:
            aucs = []
            model_indices = []
            for model in config.MODELS:
                ddf = df[(df.seq == seq) & (df.architecture == model) & (df.method == method)]
                aucs.append(ddf.auc.values[0])
                model_indices.append(config.MODEL_INDEX[model])

            data.append(dict(x=model_indices, y=aucs, name=method, mode='lines+markers',
                             marker={'symbol': config.METHOD_MARKERS[method]}))

        g = dcc.Graph(id='%s-%s-graph' % (dataset, seq), figure={
            'data': data,
            'layout': {'margin': {'l': 0, 'r': 0, 't': 0, 'b': 70}, 'showlegend': True,
                       'xaxis': dict(title='Architecture', showticklabels=True, tickangle=45, tickvals=[0,1,2,3],
                                     ticktext=[config.MODEL_NICKNAMES[m] for m in config.MODELS]),
                       'yaxis': dict(title='AUC MoRF')}
        })

        h = html.H4('SEQ-%d' % seq)

        return html.Div([h, g], style={'width': '33%', 'float': 'left'})

    graphs = []
    for d in ['mnist', 'fashion-mnist']:
        graphs.append(html.H2(d.upper()))
        for s in config.SEQS:
            graphs.append(build_graph_dataset(d, s))

    return html.Div([
                        _get_branch_div(),
                        html.H1('AUC of Flipping Curve'),
                        html.H4('using %s flip strategy and the relevances are computed from %s' % (
                            flip_function, ref_model))
                    ] + graphs)
