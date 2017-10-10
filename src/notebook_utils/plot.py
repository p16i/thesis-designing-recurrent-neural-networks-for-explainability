import logging
import matplotlib.pyplot as plt
import pandas as pd

from prettytable import PrettyTable


from utils import logging as lg

lg.set_logging()

# relative path from notebook dir
FIGURE_PATH = '../figures/nb_figures'


def setup():
    logging.info('Setup plot parameters')
    plt.show_and_save = show_and_save


def show_and_save(title=""):
    if title:
        path = '%s/%s' % (FIGURE_PATH, title)
        logging.info('save fig to %s' % path)
        plt.savefig(path)
    plt.show()


def tabularize_params(dict_params):
    '''
    :param dict_params:
    :return: Pandas dataframe with key, value columns
    '''
    table = PrettyTable()
    table._set_field_names(['attribute', 'value'])
    table.align = 'l'

    for k in sorted(dict_params):
        table.add_row([k, dict_params[k]])

    return table.get_string()


