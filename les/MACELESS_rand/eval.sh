#!/bin/bash
export CUDA_VISIBLE_DEVICES="1,3,0"

python3 ./eval_configs.py \
    --configs="odac25_xyz/odac25_rep_nos_training.extxyz" \
    --model="ODAC_RAND_L_nos_stagetwo.model" \
    --default_dtype="float32" \
    --output="odac25_xyz/odac25_xyz_training_pred_nos_large.xyz" \
    --device=cuda \
    --batch_size=1 \
