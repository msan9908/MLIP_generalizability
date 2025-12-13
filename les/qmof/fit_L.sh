#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1"

python3 ./run_train.py \
    --name="qmof" \
    --train_file="../qmof_xyz/qmof_allatoms_noswap_training.extxyz" \
    --valid_file="../qmof_xyz/qmof_allatoms_noswap_validation.extxyz" \
    --test_file="../qmof_xyz/qmof_allatoms_noswap_testing.extxyz"  \
    --energy_key="energy" \
    --E0s='average' \
    --model="MACELES" \
    --num_interactions=2 \
    --num_channels=64 \
    --hidden_irreps='64x0e + 64x1o' \
    --max_L=2 \
    --correlation=3 \
    --r_max=4.5 \
    --batch_size=4 \
    --max_num_epochs=100 \
    --forces_weight=0 \
    --energy_key="energy" \
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
