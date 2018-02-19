import fire
import logging
from utils import logging as lg, experiment_artifact

from distutils.dir_util import copy_tree
from model import provider

lg.set_logging()


MINIMUM_ACCURACY = {
    'mnist': 0.975,
    'fashion-mnist': 0.85,
    'mnist-3-digits': 0.98,
    'fashion-mnist-3-items': 0.85,
    'mnist-3-digits-maj': 0.975,
    'fashion-mnist-3-items-maj': 0.85
}

DIR_PATH='./final-models'

def main(path):
    artifact = experiment_artifact.get_result(path)

    dest_dir = '%s/%s-%s-seq-%d' % (DIR_PATH, artifact.architecture_name, artifact.dataset, artifact.seq_length)
    logging.info('Copying %s to %s' % (path, dest_dir))

    assert artifact.epoch >= 100, 'Final model should be trained at least 100 epoches'
    assert artifact.accuracy >= MINIMUM_ACCURACY[artifact.dataset], 'Accuracy is too low'

    copy_tree(path, dest_dir)


if __name__ == '__main__':
    fire.Fire(main)