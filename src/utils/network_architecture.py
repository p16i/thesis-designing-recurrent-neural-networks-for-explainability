import re


def parse(s):
    layers = map(lambda x: x.split(':'), filter(lambda x: x, re.split('[\|(\-+)]', s)))
    return dict([(k, parse_value(v)) for k, v in layers])


def parse_value(s):
    if re.match('^\d+$', s):
        return int(s)
    else:
        conv, pooling = s.split('=>')

        conv = list(map(lambda x: int(x), conv.split('x')))
        pooling = list(map(lambda x: int(x), filter(lambda x: x, re.split('[x\[\],]', pooling))))

        return {
            'conv': {
                'kernel_size': conv[:2],
                'filters': conv[-1],
            },
            'pooling': {
                'kernel_size': pooling[:2],
                'strides': pooling[-2:]
            }
        }
