import logging
from utils import logging as lg
from model import s2_network, s3_network, deep_4l_network, convdeep_4l_network
from utils import experiment_artifact

lg.set_logging()

def load(path):
    logging.debug('Load network from %s' % path)
    artifact = experiment_artifact.get_result(path)

    logging.info(artifact)

    model_loaders = {
        's2_network': s2_network.Network,
        's3_network': s3_network.Network,
        'deep_4l_network': deep_4l_network.Network,
        'convdeep_4l_network': convdeep_4l_network.Network
    }

    return model_loaders[artifact.architecture_name](artifact)

