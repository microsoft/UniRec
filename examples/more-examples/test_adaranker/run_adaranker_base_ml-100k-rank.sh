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
DATASET_NAME="ml-100k-rank"
verbose=2
learning_rate=0.001
epochs=100
n_sample_neg_train=0  #400
history_mask_mode='autoregressive'
batch_size=1024
early_stop=5

train_type="Base"
base_model="SASRec"  # GRU
max_seq_len=20
embedding_size=64
dropout_prob=0.4
n_layers=3
n_heads=8
inner_size=64
hidden_dropout_prob=0.5
attn_dropout_prob=0.5
hidden_act='gelu'
layer_norm_eps=$(awk 'BEGIN {print 1e-12}')
freeze=0

loss_type='bce'
group_size=-1
metrics="['auc','group_auc']"
key_metric="group_auc"

use_pre_item_emb=1
item_emb_path="/home/v-lichengpan/.unirec/dataset/ml-100k-rank/item_emb_64.txt"

use_wandb=0
wandb_file=""


cd $MY_DIR
export PYTHONPATH=$PWD

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
    --base_model=$base_model \
    --n_layers=$n_layers \
    --n_heads=$n_heads \
    --inner_size=$inner_size \
    --hidden_dropout_prob=$hidden_dropout_prob \
    --attn_dropout_prob=$attn_dropout_prob \
    --hidden_act=$hidden_act \
    --layer_norm_eps=$layer_norm_eps