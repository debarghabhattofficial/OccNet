#!/bin/bash

GPU_NODE=0
CONFIG="./projects/configs/bevformer/bevformer_base_occ.py"
CHECKPOINT="./work_dirs/bevformer_base_occ-2024_04_10-13_00/epoch_20.pth"
RESULTS_DIR="./results/bevformer_base_occ-2024_04_10-13_00/epoch_20"


# Following is the command executed by DEB for single_gpu_test().
# =========================================================================
srun --gres=gpu:ada1:1 \
    --mem-per-gpu=gpu_mem:6000 \
    --cpus-per-task=2 \
python "./tools/test.py" \
    $CONFIG \
    $CHECKPOINT \
    --show-dir $RESULTS_DIR \
    --eval bbox
# ==========================================================================