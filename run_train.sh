#!/usr/bin/env bash
# Fine-tune IRRA on CUHK-PEDES (single GPU)
# Usage: bash run_train.sh

DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --name irra \
  --img_aug \
  --batch_size 64 \
  --MLM \
  --dataset_name ${DATASET_NAME} \
  --root_dir ./data \
  --loss_names 'sdm+mlm+id' \
  --num_epoch 60 \
  --output_dir ./logs
