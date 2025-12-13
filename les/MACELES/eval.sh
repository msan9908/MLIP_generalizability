#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

python3 ./eval_configs.py \
    --configs="./odac25_xyz/odac25_min_rep_training.extxyz" \
    --model="ODAC_stagetwo.model" \
    --default_dtype="float32" \
    --output="./odac25_xyz/odac25_min_rep_training_maceless.xyz" \
    --device=cuda \
    --batch_size=1 \
