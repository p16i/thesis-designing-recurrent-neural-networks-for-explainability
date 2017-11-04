import logging
from utils import logging as lg
from model import s2_network, s3_network
from utils import experiment_artifact

lg.set_logging()

def load(path):
    logging.debug('Load network from %s' % path)
    artifact = experiment_artifact.get_result(path)

    logging.info(artifact)

    MODEL_LOADER = {
        's2_network': s2_network.S2Network,
        's3_network': s3_network.S3Network,
    }


    return MODEL_LOADER[artifact.architecture_name](artifact)

