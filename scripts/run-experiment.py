import tensorflow as tf
from sklearn.model_selection import ParameterGrid

from rnn_network import RNNNetwork

from model import s3_network
from model import s2_network


def run():

    result_dir = './experiment-results/sprint-4'

    experiments = [
        {
            'train': s3_network.S3Network.train,
            'param_set': [
                        # {
                        #     'seq_length': 7,  # 4 columns at a time (28)
                        #     'params': {
                        #         'architecture_str': [
                        #             'in1:100|hidden:100|out1:75|out2:10--recur:100',
                        #             'in1:100|hidden:100|out1:75|out2:10--recur:50',
                        #             'in1:100|hidden:100|out1:75|out2:10--recur:10',
                        #         ],
                        #         'lr': [0.0025, 0.005],
                        #         'keep_prob': [0.5, 0.8, 1]
                        #     }
                        # },
                        # {
                        #     'seq_length': 14,  # 2 columns at a time (28)
                        #     'params': {
                        #         'architecture_str': [
                        #             'in1:50|hidden:100|out1:75|out2:10--recur:100',
                        #             'in1:50|hidden:100|out1:75|out2:10--recur:50',
                        #             'in1:50|hidden:100|out1:75|out2:10--recur:10',
                        #         ],
                        #         'lr': [0.0025],
                        #         'keep_prob': [0.8, 1]
                        #     }
                        # },
                        # {
                        #     'seq_length': 28,  # 1 columns at a time (28)
                        #     'params': {
                        #         'architecture_str': [
                        #             'in1:20|hidden:100|out1:75|out2:10--recur:100',
                        #             'in1:20|hidden:100|out1:75|out2:10--recur:50',
                        #             'in1:20|hidden:100|out1:75|out2:10--recur:10',
                        #         ],
                        #         'lr': [0.0025],
                        #         'keep_prob': [1]
                        #     }
                        # },
                        ]
        },
        {
            'train': s2_network.S2Network.train,
            'param_set': [
                # {
                #     'seq_length': 7,  # 4 columns at a time (28)
                #     'params': {
                #         'architecture_str': [
                #             'hidden:100|out:10--recur:100',
                #             'hidden:100|out:10--recur:50',
                #             'hidden:100|out:10--recur:10',
                #         ],
                #         'lr': [0.0025],
                #         'keep_prob': [0.8, 1]
                #     }
                # },
                # {
                #     'seq_length': 14,
                #     'params': {
                #         'architecture_str': [
                #             'hidden:50|out:10--recur:100',
                #             'hidden:50|out:10--recur:50',
                #             'hidden:50|out:10--recur:10',
                #         ],
                #         'lr': [0.0025],
                #         'keep_prob': [0.8, 1]
                #     }
                # },
                {
                    'seq_length': 28,
                    'params': {
                        'architecture_str': [
                            'hidden:150|out:10--recur:100',
                            'hidden:150|out:10--recur:50',
                            'hidden:150|out:10--recur:10',
                        ],
                        'lr': [0.001],
                        'keep_prob': [1]
                    }
                },
            ]
        },


    ]

    for e in experiments:
        for s in e['param_set']:
            param_grid = ParameterGrid(s['params'])
            total = len(param_grid)
            count = 1
            for pp in param_grid:
                params = dict(seq_length=s['seq_length'], output_dir=result_dir, epoch=50, **pp)
                print('%2d/%d : %s' % (count, total, params))

                tf.reset_default_graph()
                e['train'](**params)
                count = count + 1




if __name__ == '__main__':
    run()
