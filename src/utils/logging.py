import logging

LOG_FORMAT = '%(asctime)s | %(levelname)s : %(filename)s(%(funcName)s %(lineno)d) - %(message)s'
DEFAULT_LEVEL = logging.DEBUG


def set_logging(level=DEFAULT_LEVEL):
    logging.basicConfig(format=LOG_FORMAT, level=level)

