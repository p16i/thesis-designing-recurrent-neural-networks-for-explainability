import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# relative path from notebook dir
FIGURE_PATH = '../figures/nb_figures'


def setup(plt):
    logger.info('Setup plot parameters')
    plt.show_and_save = show_and_save


def show_and_save(title=""):
    if title:
        path = '%s/%s' % (FIGURE_PATH, title)
        logger.info('save fig to %s' % path)
        plt.savefig(path)
    plt.show()
