#!/bin/bash


# This script is used to test the BEVFormer model on the OCC dataset.

# BEVFormer-based OccNet Model Naming Convention:
# ==================================================
# BEVFormer (Base): "./work_dirs/bevformer_base_occ-2024_04_10-13_00/epoch_24.pth"
# BEVFormer (Small): "./work_dirs/bevformer_small_occ-2024_05_20-20_45/epoch_24.pth"
# ==================================================
GPU_NODE=0
CONFIG="./projects/configs/bevformer/bevformer_small_occ.py"
# CHECKPOINT="./work_dirs/bevformer_base_occ/r101_dcn_fcos3d_pretrain.pth"  # ORIG
CHECKPOINT="./work_dirs/bevformer_base_occ-2024_04_10-13_00/epoch_24.pth"
RESULTS="./work_dirs/bevformer_small_occ-2024_05_20-20_45/epoch_24.pkl"


# Following is the command executed by DEB for single_gpu_test().
# =========================================================================
srun --gres=gpu:ada0:1 \
    --mem-per-gpu=gpu_mem:6000 \
    --cpus-per-task=2 \
python "./tools/test.py" \
    $CONFIG \
    --eval bbox \
    --debug

# python "./tools/test.py" \
#     $CONFIG \
#     --out $RESULTS \
#     -ckpt_path $CHEKPOINT \
#     --eval bbox \
#     --debug
# ==========================================================================