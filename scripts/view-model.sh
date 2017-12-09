#!/usr/bin/env bash

DIR=./experiment-results

PATH=`find $DIR -name "*$1*"`

/usr/bin/less "$PATH/result.yaml"