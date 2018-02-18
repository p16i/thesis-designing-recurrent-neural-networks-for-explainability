EPOCH=100

DIR=./experiment-results/sprint-9-final-model

python scripts/train.py s3_network --seq-length 1 --epoch $EPOCH --lr 0.001 --batch 50 \
        --architecture-str 'in1:512|hidden:128|out1:64|out2:10--recur:256' --keep_prob 0.5 \
            --verbose --optimizer AdamOptimizer --output_dir $DIR --dataset mnist

#python scripts/train.py deep_4l_network --seq-length 1 --epoch $EPOCH --lr 0.0001 --batch 50 --regularizer 0.0 \
#        --architecture-str 'in1:512|in2:256|hidden:128|out1:64|out2:10--recur:256' --keep_prob 0.5 --dataset mnist\
#        --verbose --output_dir $DIR

#python scripts/train.py deep_4l_network --seq-length 1 --epoch 100 --lr 0.0005 --batch 50 --regularizer 0.0 \
#        --architecture-str 'in1:512|in2:256|hidden:128|out1:64|out2:10--recur:256' --keep_prob 0.5 --dataset mnist\
#        --verbose --output_dir ./experiment-results/sprint-9-deep-4l
#
#python scripts/train.py deep_4l_network --seq-length 4 --epoch 100 --lr 0.0005 --batch 50 --regularizer 0.0 \
#        --architecture-str 'in1:512|in2:256|hidden:128|out1:64|out2:10--recur:256' --keep_prob 0.5 --dataset mnist\
#        --verbose --output_dir ./experiment-results/sprint-9-deep-4l
#
#python scripts/train.py deep_4l_network --seq-length 1 --epoch 100 --lr 0.0005 --batch 50 --regularizer 0.0 \
#        --architecture-str 'in1:512|in2:256|hidden:128|out1:64|out2:10--recur:256' --keep_prob 0.5 --dataset fashion-mnist\
#        --verbose --output_dir ./experiment-results/sprint-9-deep-4l
#
#python scripts/train.py deep_4l_network --seq-length 4 --epoch 100 --lr 0.0005 --batch 50 --regularizer 0.0 \
#        --architecture-str 'in1:512|in2:256|hidden:128|out1:64|out2:10--recur:256' --keep_prob 0.5 --dataset fashion-mnist\
#        --verbose --output_dir ./experiment-results/sprint-9-deep-4l

#python scripts/train.py convdeep_4l_network --seq-length 1 --epoch 100 --lr 0.0001 --batch 50 --regularizer 0.0 \
#        --architecture-str 'conv1:5x5x16=>2x2[2,2]|conv2:5x5x32=>2x2[2,2]|hidden:512|out1:256|out2:10--recur:256' \
#        --keep_prob 0.5 --dataset mnist\
#        --verbose --output_dir ./experiment-results/sprint-9-convdeep
#
#python scripts/train.py convdeep_4l_network --seq-length 4 --epoch 100 --lr 0.0001 --batch 50 --regularizer 0.0 \
#        --architecture-str 'conv1:5x5x16=>2x2[2,2]|conv2:5x5x32=>2x2[2,2]|hidden:512|out1:256|out2:10--recur:256' \
#        --keep_prob 0.5 --dataset mnist\
#        --verbose --output_dir ./experiment-results/sprint-9-convdeep
#
#python scripts/train.py convdeep_4l_network --seq-length 1 --epoch 100 --lr 0.0001 --batch 50 --regularizer 0.0 \
#        --architecture-str 'conv1:5x5x16=>2x2[2,2]|conv2:5x5x32=>2x2[2,2]|hidden:512|out1:256|out2:10--recur:256' \
#        --keep_prob 0.5 --dataset fashion-mnist\
#        --verbose --output_dir ./experiment-results/sprint-9-convdeep
#
#python scripts/train.py convdeep_4l_network --seq-length 4 --epoch 100 --lr 0.0001 --batch 50 --regularizer 0.0 \
#        --architecture-str 'conv1:5x5x16=>2x2[2,2]|conv2:5x5x32=>2x2[2,2]|hidden:512|out1:256|out2:10--recur:256' \
#        --keep_prob 0.5 --dataset fashion-mnist\
#        --verbose --output_dir ./experiment-results/sprint-9-convdeep
#
#python scripts/train.py s2_network --seq-length 1 --epoch $EPOCH --lr 0.0005 --batch 50 \
#            --architecture-str 'hidden:256|out:10--recur:256' --keep_prob 0.5 \
#                    --verbose --optimizer AdamOptimizer --output_dir $DIR --dataset $DATASET
#
#python scripts/train.py s2_network --seq-length 4 --epoch $EPOCH --lr 0.0005 --batch 50 \
#            --architecture-str 'hidden:256|out:10--recur:256' --keep_prob 0.5 \
#                    --verbose --optimizer AdamOptimizer --output_dir $DIR --dataset $DATASET
#
#python scripts/train.py s2_network --seq-length 7 --epoch $EPOCH --lr 0.0005 --batch 50 \
#            --architecture-str 'hidden:256|out:10--recur:256' --keep_prob 0.5 \
#                    --verbose --optimizer AdamOptimizer --output_dir $DIR --dataset $DATASET
#
#python scripts/train.py s3_network --seq-length 1 --epoch $EPOCH --lr 0.0005 --batch 50 \
#        --architecture-str 'in1:512|hidden:128|out1:64|out2:10--recur:256' --keep_prob 0.5 \
#            --verbose --optimizer AdamOptimizer --output_dir $DIR --dataset $DATASET
#
#python scripts/train.py s3_network --seq-length 4 --epoch $EPOCH --lr 0.0005 --batch 50 \
#        --architecture-str 'in1:512|hidden:128|out1:64|out2:10--recur:256' --keep_prob 0.5 \
#            --verbose --optimizer AdamOptimizer --output_dir $DIR --dataset $DATASET
#
#python scripts/train.py s3_network --seq-length 7 --epoch $EPOCH --lr 0.0005 --batch 50 \
#        --architecture-str 'in1:512|hidden:128|out1:64|out2:10--recur:256' --keep_prob 0.5 \
#            --verbose --optimizer AdamOptimizer --output_dir $DIR --dataset $DATASET

