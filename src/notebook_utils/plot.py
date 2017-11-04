import logging
import matplotlib.pyplot as plt

from utils import logging as lg

lg.set_logging()

# relative path from notebook dir
FIGURE_PATH = '../figures/nb_figures'


def setup():
    logging.debug('Setup plot parameters')
    plt.show_and_save = show_and_save


def show_and_save(title=""):
    if title:
        path = '%s/%s' % (FIGURE_PATH, title)
        logging.debug('save fig to %s' % path)
        plt.savefig(path)
    plt.show()

