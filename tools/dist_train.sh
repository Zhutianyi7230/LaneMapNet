#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28500}

CUDA_VISIBLE_DEVICES="0,1" \
CUDA_LAUNCH_BLOCKING="1" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic --autoscale-lr 
