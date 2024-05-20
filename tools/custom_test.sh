#!/bin/bash

GPU_NODE=0
CONFIG="./projects/configs/bevformer/bevformer_base_occ.py"
# CHECKPOINT="./work_dirs/bevformer_base_occ/r101_dcn_fcos3d_pretrain.pth"  # ORIG
CHECKPOINT="./work_dirs/bevformer_base_occ-2024_04_10-13_00/epoch_24.pth"
RESULTS="./results/bevformer_base_occ-2024_04_10-13_00-1/epoch_24/results.pkl"


# Following is the command executed by DEB for single_gpu_test().
# =========================================================================
srun --gres=gpu:ada0:1 \
    --mem-per-gpu=gpu_mem:6000 \
    --cpus-per-task=2 \
python "./tools/test.py" \
    $CONFIG \
    $CHECKPOINT \
    --out $RESULTS \
    --eval bbox \
    --debug
# ==========================================================================