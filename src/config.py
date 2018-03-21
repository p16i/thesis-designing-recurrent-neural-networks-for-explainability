SEQS = [1, 4, 7]
MODELS = ['s2', 's3', 'deep_4l', 'convdeep_4l']
METHODS = [
    'sensitivity',
    'guided_backprop',
    'lrp_alpha2_beta1',
    'lrp_alpha1_5_beta_5',
    'lrp_alpha1_2_beta_2',
    'lrp_deep_taylor'
]

METHOD_MARKERS = {
    'random': '0',
    'sensitivity': 'triangle-up',
    'simple_taylor': 'triangle-up-dot',
    'guided_backprop': 'square',
    'lrp_alpha2_beta1': 'star',
    'lrp_alpha3_beta2': 'star',
    'lrp_deep_taylor': 'hexagon'
}

MODEL_NICKNAMES = {
    'shallow': 'Shallow',
    'deep': 'Deep',
    'deep_v2': 'DeepV2',
    'convdeep': 'ConvDeep',
    'lenet': 'Lenet',
    'deep_v21': 'DeepV2.1',
    'rlstm': 'R-LSTM'
}

MODEL_INDEX = {
    'shallow': 0,
    'deep': 1,
    'deep_v2': 2,
    'convdeep': 3
}


def architecture_name(str):
    str = str.replace('_network', '')
    if MODEL_NICKNAMES.get(str):
        return MODEL_NICKNAMES[str]
    return str
