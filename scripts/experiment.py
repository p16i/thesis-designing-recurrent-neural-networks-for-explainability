import fire
import logging
import re
import yaml
import subprocess

from sklearn.model_selection import ParameterGrid

from utils import logging as lg

lg.set_logging(logging.INFO)

def run(config, dryrun=False, verbose=False):
    logging.info('Run experiment from %s' % config)

    with open(config, 'r') as stream:
        try:
            config_data = yaml.load(stream)
            logging.info(config_data)
        except yaml.YAMLError as exc:
            logging.error(exc)

    command_base = re.sub('[\n\\\\]', '', config_data['command'].strip())
    command_base = re.sub('\s+', ' ', command_base)
    logging.info('base command:\n%s' % command_base)

    param_grid = ParameterGrid(config_data['parameters'])
    total_experiments = len(param_grid)

    count = 1
    for p in param_grid:
        logging.info('%4d/%d : %s' % (count, total_experiments, p))
        cmd = command_base.format(**p).split(' ')
        logging.info('cmd: \n%s' % cmd)
        if not dryrun:
            subprocess.check_call(cmd)
        else:
            logging.info(cmd)

        count = count + 1


if __name__ == '__main__':
    fire.Fire(run)