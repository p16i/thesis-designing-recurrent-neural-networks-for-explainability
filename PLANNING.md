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
- [ ] Write test for LWR
- [ ] Train model with seq 1, 2, 4, 7
    - [ ] Bayesian optimization
- [ ] Implement overlapping column
- [ ] Set up /Run experiments on server (..waiting for the access)
- [ ] Save/Load model from object storage(~1 day)

Optional
- [ ] Implement result viewers (4 hours)
- [ ] Play with TFRecord (1 day)
