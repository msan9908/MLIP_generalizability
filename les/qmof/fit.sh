#!/bin/bash
export CUDA_VISIBLE_DEVICES="2"

python3 ./run_train.py \
    --name="ODAC_rand" \
    --train_file="./odac25_xyz/odac25_rep_training.extxyz" \
    --valid_file="./odac25_xyz/odac25_rep_validation.extxyz" \
    --test_file="./odac25_xyz/odac25_rep_testing.extxyz"  \
    --energy_key="energy" \
    --forces_key="forces" \
    --E0s='average' \
    --model="MACELES" \
    --num_channels=64 \
    --hidden_irreps='64x0e + 64x1o' \
    --r_max=5 \
    --max_L=1 \
    --num_interactions=2 \
    --correlation=3 \
    --batch_size=10 \
    --max_num_epochs=150 \
    --forces_weight=10000 \
    --energy_key="energy" \
    --forces_key="forces" \
    --energy_weight=10 \
    --stage_two \
    --start_stage_two=50 \
    --ema \
    --ema_decay=0.99 \
    --scheduler_patience=15 \
    --patience=50 \
    --eval_interval=1 \
    --swa \
    --amsgrad \
    --device=cuda \
    --error_table='PerAtomMAE' \
    --default_dtype="float32"\
