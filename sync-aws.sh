#!/usr/local/bin/bash

echo "syncing from $1:$2 -> local:$3"

rsync -r --progress  ec2-user@$1:/data/thesis/experiment-results/$2/* ./experiment-results/$3
