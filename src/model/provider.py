import logging
from utils import logging as lg
from model import s2_network, s3_network, deep_4l_network, convdeep_4l_network,\
    tutorial_network, shallow_2_levels, deep_v21_network, convdeep_gated, convdeep_with_mark, lstm, deep_with_sparsity,\
    shallow_v2, convdeep_output_from_rr
from utils import experiment_artifact

lg.set_logging()

MODEL_CLASS = {
    's2_network': s2_network,
    's3_network': s3_network,
    'deep_4l_network': deep_4l_network,
    'convdeep_4l_network': convdeep_4l_network,
    'tutorial_network':  tutorial_network,
    'shallow_2_levels': shallow_2_levels,
    'deep_v21_network': deep_v21_network,
    'convdeep_gated_network': convdeep_gated,
    'convdeep_with_mark': convdeep_with_mark,
    'deep_with_sparsity': deep_with_sparsity,
    'shallow_v2': shallow_v2,
    'convdeep_output_from_rr': convdeep_output_from_rr,
    'lstm': lstm
}


def load(path):
    logging.debug('Load network from %s' % path)
    artifact = experiment_artifact.get_result(path)

    logging.info(artifact)

    return MODEL_CLASS[artifact.architecture_name].Network(artifact)


def _model_path(network, dataset, seq):
    return './final-models/%s-%s-seq-%d' % (network, dataset, seq)
