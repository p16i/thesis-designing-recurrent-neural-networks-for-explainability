#!/usr/local/bin/bash

echo "syncing from $1 to $2"

rsync -r --progress  ec2-user@$1:/home/ec2-user/thesis-designing-recurrent-neural-networks-for-explainability/experiment-results/aws-training ./experiment-results/$2
