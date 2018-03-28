MAX_RELEVANCE_PERCENTAGE_PER_SAMPLE = 0.8

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
    'convdeep_transcribe':  'Conv$^+$Deep',
    'convrlstm_persisted_dropout': 'ConvR-LSTM-SD',
    'convtran_rlstm_persisted_dropout': 'Conv$^+$R-LSTM-SD',
    'conv_transribe_rlstm': 'Conv$^+$R-LSTM-SD',
    'lenet': 'Lenet',
    'deep_v21': 'DeepV2.1',
    'rlstm': 'R-LSTM',
    'rlstm_persisted_dropout': 'R-LSTM-SD',
    'deep_persisted_dropout': 'Deep-SD',
    'deep_do_xh': 'Deep$^+$'
}

MODEL_INDEX = {
    'shallow': 0,
    'deep': 1,
    'deep_v2': 2,
    'convdeep': 3
}

METHOD_ABBREVATIONS = {
    'sensitivity': 'SA',
    'guided_backprop': 'GB',
    'lrp_deep_taylor': 'DTD',
    'lrp_alpha2_beta1': '$LRP_{\\alpha_2\\beta_1}$',
    'lrp_alpha1_5_beta_5': '$LRP_{\\alpha_{1.5}\\beta_{0.5}}$',
    'lrp_alpha1_2_beta_2': '$LRP_{\\alpha_{1.2}\\beta_{0.2}}$',
}

def architecture_name(str):
    str = str.replace('_network', '')
    if MODEL_NICKNAMES.get(str):
        return MODEL_NICKNAMES[str]
    return str
