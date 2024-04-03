#!/usr/bin/env bash

rm -rf ~/.cache/torch_extensions

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

# FOllowing is the original command.
# ===============================================
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
# ===============================================

# Following is the command executed by DEB for single_gpu_test().
# =========================================================================
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval bbox
# ==========================================================================