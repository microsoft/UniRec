# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
LOCAL_ROOT="$HOME/workspace/UniRec"  # path to UniRec
###############################################################################################


# default parameters for local run
MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"


MODEL_NAME='BST'
DATA_TYPE='SeqRecDataset'
DATASET_NAME="Beauty-rank"
verbose=2
learning_rate=0.0002
epochs=100
weight_decay=0 #1e-6
dropout_prob=0
n_sample_neg_train=0  #400
max_seq_len=7
history_mask_mode='autoregressive'
embedding_size=80
batch_size=1024

# loss_type='softmax'
loss_type='bce'
# group_size=21
group_size=-1
metrics="['auc','group_auc']"
key_metric="auc"

# metrics="['hit@10;20;100', 'ndcg@10;20;100','mrr@10;20;100']"
# key_metric="mrr@100"


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
    --use_pre_item_emb=0 \
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
    --weight_decay=$weight_decay \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq"  \
    --history_mask_mode=$history_mask_mode \
    --group_size=$group_size \
    --metrics=$metrics \
    --key_metric=$key_metric \
    --shuffle_train=1 \
    --seed=2023 \
    --early_stop=5 \
    --embedding_size=$embedding_size \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --neg_by_pop_alpha=0 \
    --hidden_dropout_prob=0.4654155845792869 \
    --attn_dropout_prob=0.24153327803951888 \
    --scheduler_factor=0.5
# done
# done