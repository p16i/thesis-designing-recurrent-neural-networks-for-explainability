from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc

from pandas_datareader import data as web


from datetime import datetime as dt

import pandas as pd
import pickle

import os
cwd = os.getcwd()

import config
import  subprocess
import numpy as np

from app import app

# df = web.DataReader(
#         'COKE',
#         'yahoo',
#         dt(2017, 1, 1),
#         dt.now()
#     )
#
#
# df

# dataset = 'mnist'
# flip_function = 'minus_one'



def build_graph(df, seq, model, methods=config.METHODS):
    print(seq, model)

    data = []
    min_rel = 0
    for m in methods:
        ddf = df[(df.seq == seq) & (df.architecture == model) & (df.method == m)]
        # print([min_rel] + ddf.relevance.values)
        min_rel = np.min([min_rel, np.min(ddf.relevance)])

    print('min for %s %d : %.4f' % (model, seq, min_rel))

    for m in methods:
        ddf = df[(df.seq == seq) & (df.architecture == model) & (df.method == m)]
        relevances = ddf.relevance.values
        print(relevances)
        name = '%s(auc=%.2f)' % (m, np.trapz(relevances))
        data.append(dict(x=ddf.k, y=ddf.relevance, name=name, mode='lines+markers',
                         marker={'symbol': config.METHOD_MARKERS[m]}))
    h = html.H4('%s' % (config.MODEL_NICKNAMES[model]))
    name = "auc-graph-%s-seq-%d" % (model, seq)
    g = dcc.Graph(id='%s-graph' % name, figure={
                  'data': data,
                  'layout': {'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}, 'showlegend': True}
              })

    return html.Div([h, g], id=name, style={'width': '50%', 'float': 'left'})

#     {'x': df.index, 'y': df.Close, 'name': 'ABC Corp'},
#     {'x': df.index, 'y': df.Close+10, 'name': 'XYX Corp'}
# ],

def create_layout(dataset, flip_function):
    file = "%s/stats/aopc-%s-using-%s-flip.pkl" % (cwd, dataset, flip_function)
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
    #
        graphs = graphs + gs
    #
    #     # graphs=
    # print(graphs)

    branch = subprocess.check_output('git branch  | grep "*"', shell=True).strip().decode("utf-8")
    print(branch)

    return html.Div([
            html.Div('branch : %s' % branch.split(' ')[1]),
            html.H1('Relevance Perturbation Curve %s' % (dataset.upper())),
            html.H4('using %s flip strategy' % flip_function)
        ] + graphs)


# @app.callback(
#     Output('app-1-display-value', 'children'),
#     [Input('app-1-dropdown', 'value')])
# def display_value(value):
#     return 'You have selected "{}"'.format(value)