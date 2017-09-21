import logging
import sys

LOG_FORMAT = '%(asctime)s | %(levelname)s : %(filename)s(%(funcName)s %(lineno)d) - %(message)s'
LEVEL = logging.INFO


def set_logging():
    logging.basicConfig(format=LOG_FORMAT, level=LEVEL)

