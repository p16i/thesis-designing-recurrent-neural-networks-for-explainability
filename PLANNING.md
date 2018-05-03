## Sprint 2 : 11/09/2017 - 25/09/2017
- [x] Travis CI + Unitests
- [x] Figure Generator Utils (save output to dirs that latex can use )
- [x] Train & Test model local
- [x] Simple network diagram ( v1 )
- [x] Run v1 several times
- [x] Next improvement...

#### Sprint Review
- Always log loss and training accuracy to make sure that the network is actually learning.
- MUST not softmax before softmax_entropy layer
- Always normalize input value. ie. for minist simply divide by 255


## Sprint 3 : 25/09/2017 - 12/10/2017
- [x] v2 network 
- [x] train k columns at a time
- [x] AdamOptimizer
- [x] ask for server access
    - [ ] waiting for reply
- [x] additional layer between (input and cell
- [x] additional layer between (cell and output)
- [x] port code to Python 
- [x] Experiment with different values of seq_length
    - seq_length, 7, 14, 28
- [x] Prepare a weekly presentation
    - [x] update network figure
    - [x] write summary


## Sprint 4 : 12-19/10/2017
- [x] Save/Load model artifact
- [x] Run experiments with the shallow network(s2) (1 day)
- [X] LWR Implementation (1 day))
    - Network Class
        - Input/Output
        - Main Parameter
        - DAG
        - LWR
           
- [x] Dropout apply (2 hours)
- [x] Set experiment for s2,s3 network
    - s2:
        - seq_length: [7, 14, 28]
        - recurr: [10,50,100]
    - s3
        - seq_length: [7, 14, 28]
        - recurr: [7,15,30]
- [x] Split train, validate and test data (1 hour)

## Sprint 5 : 19/10/2017-04/11/2017
- [x] Find  good models for s2/s3 network ( acc > 98%)) (~ 1 day)
- [x] Visualize First layers for both network (1 days?)
    - [x] 28 cols at a time network 
- [x] Experiments
    - keep_prob ~ 0.1 and train s2, s3 with acc > 0.98
    - change optimizer to adams
    - summarize the relationship between dropout and clearity of heatmap
- [x] Make LWR work in batch

## Sprint 6 : 04-10/11/2017
- [x] Write test for LWR
- [x] Train model with 
    - seq 1, 2 4 : s2, s3
    - seq 7, 14, 28 : s3 only 
    - [x] Bayesian optimization
- [ ] ~~Implement overlapping column~~ not relevant
- [x] Set up /Run experiments on server (..waiting for the access)
- [ ] ~~Save/Load model from object storage(~1 day)~~ postpone

## Sprint 7 : 10-17/11/2017
- [x] Fix dropout
    - [x] Retrain models for seq-4,7,14
- [x] Relevance distribution for each class and per architecture
    - seq 4, 7, 14
- [x] Sensitivity, Simple Taylor
- [x] Proportion / positive and negative weights 
- [x] Prepare meeting note
    - [x] relevance distributions from lrp
- [x] New propagation rules

## Sprint 8 : 17/11-12/12/2017
- [x] boxplot weight visualization
- [x] Train 500 units for seq 4, 7 with 100 epochs
    - no dropout at recurrent
- [x] Prepare report structure
- [x] Fashin MNIST https://github.com/zalandoresearch/fashion-mnist
    - write data wrapper
    - s2
    - experiment with s2/s3

## Sprint 9 : 15/12/2017-01/01/2018
- [x] Convolutional models
- [x] FasionMNIST acc> 90%
- [x] Tensorboard
- [x] UFI Dataset(http://ufi.kiv.zcu.cz/)
- v0.1 report
    - chapter experiment
    

## Sprint 10-11 : 1-12/2018
## Sprint 12 : 12-19/01/2018
## Sprint 13 : 19-26/01/2018
- [x] dash visualization
    - AUC
- [x] train with artificial class to force relevance positive
    - conv-fashion1 : doesn't seem to work    
    - conv-mnist1 : same as above
    - [x] will bigger constant affect the explanation?
        - no, only shift relevance to 
    deep-mnist1 : 
- [x] what happen if relevance negative or zero, how heatmap -1, 0, 1 look like
    - see s13-test-explaning.ipynb
- [x] AUC curve exploration
    - [x] flip with zero
        - simple taylor much better
    - flip to minus one ( w/o pseudo class )
        -  fashion-mnist  
            guided-backprop(GB), sensitivity perform the best for seq-1 for any architecture, 
            except convdeep that LRP-deep-taylor is on par with GB
            For **seq-4**, LRP-deep-taylor is the best for Deep, DeepV2
            but worse than GB for ConvDeep
            For **seq-7**, similar to seq-4 and LRP-deep-taylor wins GB for ConvDeep
- [x] Use Sum pooling
    - seem to be worse than max pooling for MNIST, FMNIST Sum is better
- [ ] distribute pred instead of target

## Sprint 14 : 02-07/02/2018
- [x] Derived AUC from reference models
    - models
        - conv-seq1
        - conv-same-seq
        - lenet (tutorial network)
    - urls
        - http://127.0.0.1:8050/pages/auc/<REF_MODEL>/<DATASET>/minus_one
        - http://127.0.0.1:8050/pages/auc-summary/<REF_MODEL>/minus_one
    
    
## Sprint 15 : 07-21/02/2018
- [x] MNIST, FMNIST 3 digits data and train models
    - hypothesis:
        - deeper models are better to propagate relevance,
        - LRP is better? 
            - percentage distribution of positive relevance for columns 5-8th
- [x] Plot similar to auc for 3 digits
- [ ] Retrain models
- [x] Flipping Auc with negative value
    - doesn't work
- [ ] Writing something
    - [ ] Summary page
- [ ] UFI acc > 60%
- [ ] unify data classes


## Idea for long term deps
- dropout mark same for all sequence
- attention 
- 2 levels
    


Not for New AMI
- add time to ./run.py command
- add script to activate and fetch code
- install scikit-image

Optional
- [ ] continuous training
    - saving model
    - use bayesian to explore space and load the best model from bayesion opt and train futher
- [ ] Training on multiple GPUs
- [ ] Implement result viewers (4 hours)
- [ ] Play with TFRecord (1 day)
## Sprint 14 : 26/01-2/02/2018
- 80% report done
## Sprint 15 : 2-9/02/2018
## Sprint 16 : 9-16/02/2018
## Sprint 17 : 16-23/02/2018

## Sprint 18 : 23/02-2/03/2018
## Sprint 19 : 2-9/03/2018
## Sprint 20 : 9-16/03/2018
## Sprint 21 : 16-23/03/2018

## Sprint 22 : 23-30/03/2018 <- LAST ONE?


```bash
# testing code
                #======== test code
                # import numpy as np
                # total_relevance_reduced = tf.reduce_sum(self.dag.total_relevance, axis=1)
                #
                # rx = np.zeros((x.shape[0], self.architecture.recur))
                # # z = tf.matmul(self.dag.ha_l1_activations[-1], tf.minimum(0.0, self.dag.layers['output_l1'].W))
                # # s = rel_to_output_l1 / z
                # # ss = tf.matmul(s, tf.transpose(w))
                # pred, total_relevance, rr_of_pixels = sess.run(
                #     [self.dag.y_pred, total_relevance_reduced, [rel_to_input[-1], rel_to_rr_l2, rel_to_hidden_l1]],
                #     feed_dict={self.dag.x: x, self.dag.y_target: y, self.dag.rx: rx, self.dag.keep_prob: 1})
                #
                #
                # total_relevance_pixels = 0
                # for rr in rr_of_pixels:
                #     total_relevance_pixels =  total_relevance_pixels + np.sum(rr, axis=1)
                #
                # # print(z[83:86])
                #
                # diff_greater_than_threshold = (np.abs(total_relevance_pixels - total_relevance) > base.TEST_RELEVANCE_THRESHOLD)
                # no_diff_greater_than_threshold = np.sum(diff_greater_than_threshold)
                #
                # for i in range(total_relevance_pixels.shape[0]):
                #     rp = total_relevance_pixels[i]
                #     re = total_relevance[i]
                #     print('%d: out: %f \t\t | exp: %f \t\t diff %f(%s)'
                #           % (i, rp, re, rp-re, diff_greater_than_threshold[i]))
                #
                # print('there are %d diffs greather than threshold %f' % (no_diff_greater_than_threshold, base.TEST_RELEVANCE_THRESHOLD))
                # assert no_diff_greater_than_threshold < 0.05*total_relevance_pixels.shape[0], \
                #     'Conservation property isn`t hold\n : Sum of relevance from pixels is not equal to output relevance.'
                #
                # raise SystemError('Force Exist')
                #======= end test
```
