#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
GPU_NODE=0
CONFIG="./projects/configs/bevformer/bevformer_base_occ.py"
CHECKPOINT="./work_dirs/bevformer_base_occ/r101_dcn_fcos3d_pretrain.pth"


# Following is the command executed by DEB for single_gpu_test().
# =========================================================================
srun --gres=gpu:ada0:1 \
    --mem-per-gpu=gpu_mem:6000 \
    --cpus-per-task=2 \
python "./tools/test.py" \
    $CONFIG \
    $CHECKPOINT \
    --eval bbox
# ==========================================================================