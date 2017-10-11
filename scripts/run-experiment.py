from sklearn.model_selection import ParameterGrid

from rnn_network import RNNNetwork

def run():
    rnn = RNNNetwork()

    result_dir = './experiment-results/sprint-2'

    param_set = [
        # 1, 7, 14, 28
        # {
        #     'seq_length': 7, # 4 columns at a time
        #     'params': {
        #         'architecture_str': [
        #             'in1:100|hidden:60|out1:50|out2:10--recur:7',
        #             'in1:100|hidden:60|out1:30|out2:10--recur:7',
        #             'in1:100|hidden:60|out1:50|out2:10--recur:10',
        #             'in1:100|hidden:60|out1:30|out2:10--recur:10',
        #             'in1:80|hidden:60|out1:50|out2:10--recur:7',
        #             'in1:80|hidden:60|out1:30|out2:10--recur:7',
        #             'in1:80|hidden:60|out1:50|out2:10--recur:10',
        #             'in1:80|hidden:60|out1:30|out2:10--recur:10',
        #         ],
        #         'lr': [0.01, 0.005],
        #         'epoch': [10, 20, 50]
        #     }
        # },
        # {
        #     'seq_length': 14,  # 2 columns at a time ( 28*2 = 56 )
        #     'params': {
        #         'architecture_str': [
        #             'in1:50|hidden:60|out1:50|out2:10--recur:7',
        #             'in1:50|hidden:60|out1:30|out2:10--recur:7',
        #             'in1:50|hidden:60|out1:50|out2:10--recur:20',
        #             'in1:50|hidden:60|out1:30|out2:10--recur:20',
        #             'in1:30|hidden:60|out1:50|out2:10--recur:7',
        #             'in1:30|hidden:60|out1:30|out2:10--recur:7',
        #             'in1:30|hidden:60|out1:50|out2:10--recur:20',
        #             'in1:30|hidden:60|out1:30|out2:10--recur:20',
        #         ],
        #         'lr': [0.01, 0.005],
        #         'epoch': [10, 20, 50]
        #     }
        # },
        {
            'seq_length': 28,  # 1 column at a time (28)
            'params': {
                'architecture_str': [
                    'in1:20|hidden:60|out1:50|out2:10--recur:20',
                    'in1:20|hidden:60|out1:30|out2:10--recur:20',
                    'in1:20|hidden:60|out1:50|out2:10--recur:40',
                    'in1:20|hidden:60|out1:30|out2:10--recur:40',
                    'in1:10|hidden:60|out1:50|out2:10--recur:20',
                    'in1:10|hidden:60|out1:30|out2:10--recur:20',
                    'in1:10|hidden:60|out1:50|out2:10--recur:40',
                    'in1:10|hidden:60|out1:30|out2:10--recur:40',
                ],
                'lr': [0.01, 0.005],
                'epoch': [10, 20, 50]
            }
        },
    ]

    for s in param_set:
        param_grid = ParameterGrid(s['params'])
        total = len(param_grid)
        count = 1
        for pp in param_grid:
            params = dict(seq_length=s['seq_length'], result_dir=result_dir, **pp)
            print('%2d/%d : %s' % (count, total, params))

            rnn.s2_network(**params)
            count = count + 1


if __name__ == '__main__':
    run()
