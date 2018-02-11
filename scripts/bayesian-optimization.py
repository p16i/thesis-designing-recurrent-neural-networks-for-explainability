import fire
import logging
from skopt import gp_minimize

from utils import logging as lg
lg.set_logging(logging.INFO)

import train

def run(network, seq, architecture,
        batch_size=50, epoch=100,
        lr_bound=(0.0001, 0.001),
        no_experiments=100, dataset='mnist', output_dir=""):
    logging.info('Optimization for %s with seq %s' % (network, seq))
    logging.info('--------------------')
    logging.info('architecture : %s' % architecture)
    logging.info('learning rate bounds : (%s, %s)' % lr_bound)
    # logging.info('keep_prob bounds : (%s, %s)' % keep_prob_bound)
    logging.info('--------------------')
    logging.info('Experiment artifacts will be saved to %s' % output_dir)

    def objective(x):
        params = {
            'network': network,
            'seq_length': seq,
            'batch': batch_size,
            'architecture_str': architecture,
            'lr': x[0],
            'keep_prob': 0.5,
            'output_dir': output_dir,
            'epoch': epoch,
            'dataset': dataset,
            'verbose': True
        }

        logging.info('--------- PARAMS ---------')
        logging.info(params)

        artifact = train.train(**params)

        return 1-artifact.val_accuracy

    logging.info('#########################')

    res = gp_minimize(objective, [lr_bound], verbose=True, n_calls=no_experiments)

    logging.info('------- RESULT --------')
    logging.info(res)

    logging.info('------- Summary -------')
    logging.info('Best val_acc %f' % res.fun)
    logging.info('at %s' % res.x)


if __name__ == '__main__':
    fire.Fire(run)
