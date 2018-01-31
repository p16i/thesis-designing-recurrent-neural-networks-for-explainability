aws ec2 request-spot-instances  --spot-price "$1" --instance-count 1 --type "one-time" --launch-specification file://specification.json
