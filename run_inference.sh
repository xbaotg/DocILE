#!/bin/bash

# set -euo pipefail

# Set GPU device number to use, enforced with CUDA_VISIBLE_DEVICES=${GPU}
GPU="7"

# Call with `./run_inference test` or `./run_inference.sh val`
split=$1

run="UIT@AICLUB_TAB"


# The '-user' suffix is here not to overwrite downloaded predictions by mistake
PREDICTIONS_DIR_PREFIX="predictions"
mkdir -p ${PREDICTIONS_DIR_PREFIX}

DOCILE_PATH="/mlcv/WorkingSpace/Personals/thinhvq/DocILE/docile/data_train"

# Detections of tables and Line Items. Only used if options --crop_bboxes_filename and/or
# --line_item_bboxes_filename are used as well.
TABLE_TRANSFORMER_PREDICTIONS_DIR="/app/data/baselines/predictions/detr"


function run_inference() {
  cmd=$1
  split=$2
  # checkpoint_subdir=$3
  output_dir="${PREDICTIONS_DIR_PREFIX}/$3"
  shift ; shift ; shift ; shift
  extra_params=$@

  mkdir -p $output_dir

  log="${output_dir}/log_inference.txt"

  run_cmd=$(tr '\n' ' ' << EOF
CUDA_VISIBLE_DEVICES=${GPU} python ${cmd}
    --split ${split}
    --docile_path ${DOCILE_PATH}
    --output_dir ${output_dir}
    --store_intermediate_results
    --merge_strategy new
    ${extra_params} 2>&1 | tee -a ${log}
EOF
  )
  echo ${run_cmd}
  echo ${run_cmd} >> ${log}
  eval ${run_cmd}
}


run_inference inference.py $split $run