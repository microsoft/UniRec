# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
HOME_DIR=$(eval echo ~)

LOCAL_ROOT="$HOME_DIR/work/UniRec"
###############################################################################################


# default parameters for local run
MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"
cd $MY_DIR
export PYTHONPATH=$PWD

DATASET_NAME="ml-100k"
MODEL_NAME='SAR'   # SAR, UserCF, SLIM, AdmmSLIM, EASE, MultiVAE
DATA_TYPE='AERecDataset' 
test_protocol='one_vs_all'  #'one_vs_all' 'session_aware' 
user_history_filename='user_history' #'user_history'
user_history_file_format='user-item_seq' #"user-item" 
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT 
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --valid_protocol=$test_protocol \
    --test_protocol=$test_protocol \
    --history_mask_mode='autoregressive' \
    --user_history_filename=$user_history_filename \
    --user_history_file_format=$user_history_file_format \
    --metrics="['hit@10;20', 'ndcg@10;20']" \
    --key_metric="ndcg@10" \
    --verbose=2
