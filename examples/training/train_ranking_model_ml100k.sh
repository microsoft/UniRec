# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

HOME_DIR=$(eval echo ~)

LOCAL_ROOT="$HOME_DIR/work/UniRec"

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"



# default parameters for local run
MODEL_NAME='BST'
DATA_TYPE='SeqRecDataset'
DATASET_NAME="ml-100k"
verbose=2
learning_rate=0.001
epochs=100
weight_decay=0 #1e-6
dropout_prob=0
loss_type='softmax'
n_sample_neg_train=20 
max_seq_len=7
history_mask_mode='autoregressive'
embedding_size=80

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
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=$epochs  \
    --batch_size=256 \
    --n_sample_neg_train=$n_sample_neg_train \
    --n_sample_neg_valid=99 \
    --valid_protocol='one_vs_k' \
    --n_sample_neg_test=99 \
    --test_protocol='one_vs_k' \
    --grad_clip_value=10 \
    --weight_decay=$weight_decay \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq"  \
    --history_mask_mode=$history_mask_mode \
    --metrics="['hit@10;20', 'ndcg@10;20','mrr@10;20']" \
    --key_metric="ndcg@10" \
    --shuffle_train=1 \
    --seed=2023 \
    --early_stop=5 \
    --embedding_size=$embedding_size \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --neg_by_pop_alpha=0
# done