# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
HOME_DIR=$(eval echo ~)  # root
LOCAL_ROOT="$HOME_DIR/workspace/UniRec"
DATASET_NAME="ml-10m-adaranker"  # ml-100k-adaranker

pipelines=(0 1 2)  # 0: train base, 1: train ada-ranker, 2: train base + ada-ranker

embedding_size=64
item_emb_path="$HOME_DIR/.unirec/dataset/$DATASET_NAME/item_emb_$embedding_size.txt"


###############################################################################################
############################## default parameters for local run ###############################
###############################################################################################
MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"

MODEL_NAME='AdaRanker'
DATA_TYPE='SeqRecDataset'

verbose=2
learning_rate=0.001
epochs=100
dropout_prob=0.4
n_sample_neg_train=0  #400
history_mask_mode='autoregressive'
batch_size=1024
base_model="GRU"
loss_type='bce'
group_size=-1
metrics="['auc','group_auc']"
key_metric="group_auc"
early_stop=5

if [ $DATASET_NAME == "ml-100k-adaranker" ]; then
    max_seq_len=20
elif [ $DATASET_NAME == "ml-10m-adaranker" ]; then
    max_seq_len=200
fi


cd $MY_DIR
export PYTHONPATH=$PWD

for pipeline in ${pipelines[@]}; do

    if [ $pipeline -eq 0 ] || [ $pipeline -eq 2 ]; then
        use_pre_item_emb=1
        freeze=0
        train_type="Base"

        currentTime=`date "+%Y-%m-%d_%H%M%S"`
        base_checkpoint_dir="checkpoint_"$currentTime

        BASE_ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/${MODEL_NAME}_Base"
        mkdir -p $BASE_ALL_RESULTS_ROOT
        ###################### train base model ######################
        python unirec/main/main.py \
            --config_dir="unirec/config" \
            --model=$MODEL_NAME \
            --dataloader=$DATA_TYPE \
            --dataset=$DATASET_NAME \
            --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
            --output_path=$BASE_ALL_RESULTS_ROOT"/train" \
            --learning_rate=$learning_rate \
            --dropout_prob=$dropout_prob \
            --use_pre_item_emb=$use_pre_item_emb \
            --item_emb_path=$item_emb_path \
            --loss_type=$loss_type \
            --max_seq_len=$max_seq_len \
            --has_user_bias=0 \
            --has_item_bias=0 \
            --epochs=$epochs \
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
            --freeze=$freeze \
            --base_model=$base_model \
            --checkpoint_dir=$base_checkpoint_dir

        echo ">>> Base model training step completed."
    fi


    if [ $pipeline -eq 1 ] || [ $pipeline -eq 2 ]; then
        if [ $pipeline -eq 1 ]; then
            use_pre_item_emb=1
            freeze=0
            load_pretrained_model=0
            model_file=""
            ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/${MODEL_NAME}_Ada-Ranker"

        elif [ $pipeline -eq 2 ]; then
            use_pre_item_emb=0
            freeze=1
            load_pretrained_model=1
            model_file="$BASE_ALL_RESULTS_ROOT/train/$base_checkpoint_dir/$MODEL_NAME.pth"
            ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/${MODEL_NAME}_Finetune"
        fi
        train_type="Ada-Ranker"
        currentTime=`date "+%Y-%m-%d_%H%M%S"`
        checkpoint_dir="checkpoint_"$currentTime

        mkdir -p $ALL_RESULTS_ROOT
        ###################### train Ada-Ranker ######################
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
            --freeze=$freeze \
            --load_pretrained_model=$load_pretrained_model \
            --model_file=$model_file \
            --checkpoint_dir=$checkpoint_dir

        echo ">>> AdaRanker training or finetune step completed."
    fi

done