#!/bin/bash


CONFIG="./projects/configs/bevformer/bevformer_base_occ.py"
WORK_DIR="./work_dirs/bevformer_base_occ-2024_04_10-12_15/"
NUM_GPUS=1


# Following is the command executed by DEB for single_gpu_test().
# ==========================================================================
python "./tools/train.py" \
    $CONFIG \
    --work-dir $WORK_DIR \
    --gpus $NUM_GPUS
# ==========================================================================