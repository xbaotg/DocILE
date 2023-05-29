#!/bin/bash

set -euo pipefail

# Config here
GPU="4,5,6,7"
run="UIT@AICLUB_TAB"
OUTPUT_DIR_PREFIX="experiments"
DATA="--docile_path /mlcv/WorkingSpace/Personals/thinhvq/DocILE/docile/data_train"
USE_PREPROCESSED="--preprocessed_dataset_path /docile/cached_images"
CHECKPOINT="/docile/training/roberta_base_with_synthetic_pretraining/20230515_1324_47/checkpoint-3690"
OTHER_COMMON_PARAMS="--save_total_limit 5 --weight_decay 0.01 --lr 5e-6 --dataloader_num_workers 32 --use_BIO_format --tag_everything --report_all_metrics"

TIMESTAMP=$(date +"%Y%m%d_%H%M_%S")
COMMON_PARAMS="${DATA} ${USE_PREPROCESSED} ${OTHER_COMMON_PARAMS}"

# Use this when you want to want with synthetic for faster training
# USE_ARROW="--arrow_format_path /app/data/baselines/preprocessed_dataset_arrow_format"

function run_training() {
  cmd=$1
  output_dir="${OUTPUT_DIR_PREFIX}/$2"
  shift ; shift
  params="$@ --output_dir ${output_dir}"

  mkdir -p ${output_dir}
  log="${output_dir}/log_train.txt"

  training_cmd="TF_FORCE_GPU_ALLOW_GROWTH=\"true\" CUDA_VISIBLE_DEVICES=${GPU} python ${cmd} ${params} 2>&1 | tee ${log}"

  echo "-----------"
  echo "Parameters:"
  echo "-----------"
  echo ${params}
  echo "-----------"
  echo "Running ${training_cmd}"
  echo "-----------"
  echo "==========="

  eval ${training_cmd}
}

train_params="--train_bs 4 --test_bs 4 --num_epochs 300 --gradient_accumulation_steps 4"
model="--model_name ${CHECKPOINT} --use_roberta"
all_params="${COMMON_PARAMS} ${train_params} ${model}"
output_dir="${run}/${TIMESTAMP}"
run_training train.py ${output_dir} ${all_params}