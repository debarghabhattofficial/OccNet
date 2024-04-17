#!/bin/bash


GPU_NODE=0
CONFIG="./projects/configs/bevformer/bevformer_base_occ.py"
CHECKPOINT="./work_dirs/bevformer_base_occ/r101_dcn_fcos3d_pretrain.pth"


# Following is the command executed by DEB for single_gpu_test().
# =========================================================================
python "./tools/pt2onnx.py" \
    $CONFIG \
    $CHECKPOINT
# ==========================================================================