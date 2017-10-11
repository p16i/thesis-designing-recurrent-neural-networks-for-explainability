import glob
import yaml
import pandas as pd


def get_results(path):
    '''
    :param path:  path to a directory containing result files
    :return: pandas dataframe
    '''

    files = glob.glob(path + '/*.yaml')

    results = []
    for yf in files:
        with open(yf, 'r') as stream:
            try:
                res = yaml.load(stream)
                results.append(res)
            except yaml.YAMLError as exc:
                print(exc)

    return pd.DataFrame(results)
