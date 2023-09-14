#!/bin/bash

# root
LOCAL_ROOT='/home/v-leiyuxuan/working_dir/UniRec'

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="/home/v-leiyuxuan/blob/final_data/unirec_data"
OUTPUT_ROOT="$LOCAL_ROOT/output"
cd $MY_DIR
export PYTHONPATH=$PWD

DATASET_NAME="ES_final_dataset"
MODEL_NAME='UserCF'   # SAR, UserCF
DATA_TYPE='AERecDataset'  # BaseDataset SeqRecDataset
test_protocol='one_vs_all'  #'one_vs_all' 'session_aware' 
user_history_filename='user_history' #'user_history'
user_history_file_format='user-item_seq' #"user-item_seq" 
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT 
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --test_protocol=$test_protocol \
    --history_mask_mode='autoregressive' \
    --user_history_filename=$user_history_filename \
    --user_history_file_format=$user_history_file_format \
    --metrics="['hit@20;100;200;300;1000', 'ndcg@20;100;200;300;1000','mrr@20;100;200;300;1000']" \
    --key_metric="mrr@100" \
    --verbose=2
