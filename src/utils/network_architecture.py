import re


def parse_architecture(str):
    layers = re.findall('([a-z0-9]+):(\d+)', str)

    return dict([(k, int(v)) for k, v in layers])
