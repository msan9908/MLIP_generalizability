#!/bin/bash
export CUDA_VISIBLE_DEVICES="3"

python3 ./eval_configs.py \
    --configs="../qmof_xyz/qmof_allatoms_noswap_training.extxyz" \
    --model="qmof_stagetwo.model" \
    --default_dtype="float32" \
    --output="../qmof_xyz/qmof_allatoms_noswap_training_pred.xyz" \
    --device=cuda \
    --batch_size=1 \
