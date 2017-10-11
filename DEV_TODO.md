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
- [ ] Prepare a weekly presentation
    - [x] update network figure
    - [ ] write summary


## Sprint 4 : 12-19/10/2017
- [ ] Split train, validate and test data
- [ ] Run experiments on server
- [ ] Play with TFRecord
