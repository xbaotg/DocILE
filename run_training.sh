#!/bin/bash

# You can use this script to run NER baselines training. All of the trainings below use either
# publicly available checkpoints (e.g., roberta-base or microsoft/layoutlmv3-base) or checkpoints
# that can be produced with this script (some of them are provided with the docile dataset). To run
# from your checkpoint instead, change the `--model_name` parameter. E.g., to train
# "roberta_ours_with_synthetic_pretraining" from scratch, you should
#  * run "roberta_pretraining",
#  * change the path of `--model_name` in "roberta_ours_synthetic_pretraining" and run it,
#  * change the path of `--model_name` in "roberta_ours_with_synthetic_pretraining" and run it.

set -euo pipefail

# Set GPU device number to use, enforced with CUDA_VISIBLE_DEVICES=${GPU}
GPU="4,5,6,7"
run="UIT@AICLUB_TAB"
TIMESTAMP=$(date +"%Y%m%d_%H%M_%S")
OUTPUT_DIR_PREFIX="experiments"
DATA="--docile_path /mlcv/WorkingSpace/Personals/thinhvq/DocILE/docile/data_train"
USE_PREPROCESSED="--preprocessed_dataset_path /docile/cached_images"
OTHER_COMMON_PARAMS="--save_total_limit 5 --weight_decay 0.01 --lr 5e-6 --dataloader_num_workers 32 --use_BIO_format --tag_everything --report_all_metrics"
COMMON_PARAMS="${DATA} ${USE_PREPROCESSED} ${OTHER_COMMON_PARAMS}"

# Used for synthetic pretraining of LayoutLMv3
USE_ARROW="--arrow_format_path /app/data/baselines/preprocessed_dataset_arrow_format"


function run_training() {
  cmd=$1
  output_dir="${OUTPUT_DIR_PREFIX}/$2"
  shift ; shift
  params="$@ --output_dir ${output_dir}"

  mkdir -p ${output_dir}
  log="${output_dir}/log_train.txt"

  training_cmd="TF_FORCE_GPU_ALLOW_GROWTH=\"true\" CUDA_VISIBLE_DEVICES=${GPU} poetry run python ${cmd} ${params} 2>&1 | tee ${log}"

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
model="--model_name /docile/training/roberta_base_with_synthetic_pretraining/20230515_1324_47/checkpoint-3690 --use_roberta"
all_params="${COMMON_PARAMS} ${train_params} ${model}"
output_dir="${run}/${TIMESTAMP}"
run_training train.py ${output_dir} ${all_params}