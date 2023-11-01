# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

# root
HOME_DIR=$(eval echo ~)
LOCAL_ROOT="$HOME_DIR/workspace/microsoft/UniRec"

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"


# default parameters for local run
MODEL_NAME='AdaRanker'
DATA_TYPE='SeqRecDataset'
# DATASET_NAME="ml-10m-rank"
DATASET_NAME="ml-100k-rank"
verbose=2
learning_rate=0.001
epochs=100
dropout_prob=0.4
n_sample_neg_train=0  #400
max_seq_len=20
history_mask_mode='autoregressive'
embedding_size=64
batch_size=1024

# loss_type='softmax'
loss_type='bce'
group_size=-1
metrics="['auc','group_auc']"
key_metric="group_auc"
train_type="Ada-Ranker"
base_model="GRU"

use_pre_item_emb=1
# item_emb_path="/home/v-lichengpan/workspace/microsoft/UniRec/data/ml-10m-rank/item_emb_64.txt"
item_emb_path="/home/v-lichengpan/.unirec/dataset/ml-100k-rank/item_emb_64.txt"

# metrics="['hit@10;20;100', 'ndcg@10;20;100','mrr@10;20;100']"
# key_metric="mrr@100"

use_wandb=0
wandb_file=""
freeze=0
early_stop=5


cd $MY_DIR
export PYTHONPATH=$PWD

# for loss_type in 'softmax' 'bce'
# do
# for use_features in 1 0
# do
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT
### train ###################################
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --learning_rate=$learning_rate \
    --dropout_prob=$dropout_prob \
    --use_pre_item_emb=$use_pre_item_emb \
    --item_emb_path=$item_emb_path \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=$epochs  \
    --batch_size=$batch_size \
    --n_sample_neg_train=$n_sample_neg_train \
    --n_sample_neg_valid=0 \
    --valid_protocol='one_vs_k' \
    --test_protocol='one_vs_k' \
    --grad_clip_value=10 \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --history_mask_mode=$history_mask_mode \
    --group_size=$group_size \
    --metrics=$metrics \
    --key_metric=$key_metric \
    --shuffle_train=1 \
    --seed=2023 \
    --early_stop=$early_stop \
    --embedding_size=$embedding_size \
    --train_type=$train_type \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --neg_by_pop_alpha=0 \
    --scheduler_factor=0.5 \
    --use_wandb=$use_wandb \
    --wandb_file=$wandb_file \
    --freeze=$freeze \
    --base_model=$base_model
# done
# done