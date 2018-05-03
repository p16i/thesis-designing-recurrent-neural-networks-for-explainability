import logging

import model.architectures
from utils import experiment_artifact
from utils import logging as lg

lg.set_logging()


def get_architecture_class(arch):
    return getattr(model.architectures, arch)


def load(path):
    logging.debug('Load network from %s' % path)
    artifact = experiment_artifact.get_result(path)

    logging.info(artifact)

    return get_architecture_class(artifact.architecture_name).Network(artifact)


def _model_path(network, dataset, seq):
    return './final-models/%s-%s-seq-%d' % (network, dataset, seq)
