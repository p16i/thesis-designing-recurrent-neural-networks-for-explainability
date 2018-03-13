SEQS = [1, 4, 7]
MODELS = ['s2', 's3', 'deep_4l', 'convdeep_4l']
METHODS = [
    'sensitivity',
    'guided_backprop',
    'lrp_alpha2_beta1',
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
    's2': 'Shallow',
    'shallow': 'Shallow',
    's3': 'Deep',
    'deep': 'Deep',
    'deep_4l': 'DeepV2',
    'deep_v2': 'DeepV2',
    'convdeep_4l': 'ConvDeep',
    'convdeep': 'ConvDeep',
    'tutorial': 'lenet',
    'shallow_2_levels': 'shallow_2_levels',
    'deep_v21': 'DeepV2.1',
    'convdeep_gated': 'convdeep_gated'
}

MODEL_INDEX = {
    's2': 0,
    's3': 1,
    'deep_4l': 2,
    'convdeep_4l': 3
}


def architecture_name(str):
    str = str.replace('_network', '')
    if MODEL_NICKNAMES.get(str):
        return MODEL_NICKNAMES[str]
    return str
