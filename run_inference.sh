#!/bin/bash

# set -euo pipefail

# Config here

# Set GPU device number to use, enforced with CUDA_VISIBLE_DEVICES=${GPU}
GPU="6"
run="UIT@AICLUB_TAB"
PREDICTIONS_DIR_PREFIX="predictions"
DOCILE_PATH="/mlcv/WorkingSpace/Personals/thinhvq/DocILE/docile/data_train"

# Call with `./run_inference test` or `./run_inference.sh val`
split=$1

# The '-user' suffix is here not to overwrite downloaded predictions by mistake
mkdir -p ${PREDICTIONS_DIR_PREFIX}


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


# To config checkpoint, you need to config it in config.py

run_inference inference.py $split $run