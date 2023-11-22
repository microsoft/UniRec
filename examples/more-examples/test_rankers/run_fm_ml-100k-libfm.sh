# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
HOME_DIR=$(eval echo ~)
LOCAL_ROOT='$HOME_DIR/workspace/UniRec'

# task="test"
# model_file="$HOME_DIR/workspace/UniRec/output/Beauty/FM/train/checkpoint_2023-08-28_063744_32/FM.pth"
# model_file="$HOME_DIR/workspace/UniRec/output/Beauty/FM/train_pre/xlearn-ckpt_20230822/FM.txt"

task="train"
model_file=""


###############################################################################################
############################## default parameters for local run ###############################
###############################################################################################
MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"


MODEL_NAME='FM'
DATA_TYPE='RankDataset'
DATASET_NAME="ml-100k-libfm"
verbose=2
epochs=500
n_sample_neg_train=0  #400
history_mask_mode='autoregressive'
embedding_size=80
linear_mode='gather'
score_clip_value=-1
loss_type='bce'

# learning_rate=0.01
# batch_size=8
# group_size=-1
# metrics="['auc']"
# key_metric="auc"
# optimizer="adagrad"

learning_rate=0.001
batch_size=1024
group_size=21
metrics="['group_auc','auc']"
key_metric="auc"
optimizer="adam"


cd $MY_DIR
export PYTHONPATH=$PWD


# for learning_rate in 0.02 0.01 0.005 0.002 0.001 0.0005; do
#     for batch_size in 64 128 256 512 1024; do
#         for optimizer in "adam" "adagrad"; do
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT
### train ###################################
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/"$task \
    --learning_rate=$learning_rate \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
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
    --scheduler_factor=0.5 \
    --linear_mode=$linear_mode \
    --score_clip_value=$score_clip_value \
    --task=$task \
    --model_file=$model_file \
    --optimizer=$optimizer
#         done
#     done
# done