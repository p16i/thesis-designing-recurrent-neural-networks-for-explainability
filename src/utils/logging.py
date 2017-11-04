import logging

from prettytable import PrettyTable


LOG_FORMAT = '%(asctime)s | %(levelname)s : %(filename)s(%(funcName)s %(lineno)d) - %(message)s'
DEFAULT_LEVEL = logging.DEBUG


def set_logging(level=DEFAULT_LEVEL):
    logging.basicConfig(format=LOG_FORMAT, level=level)

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
