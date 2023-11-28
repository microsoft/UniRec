# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash
# get scores of a set of items for every user in valid set and test set,  and save the scores in a file

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
LOCAL_ROOT="$HOME/workspace/UniRec"  # path to UniRec

ALL_DATA_ROOT="$HOME/blob/final_data/unirec_data"

model_file="$HOME/blob/output/ES_final_dataset/Avghist/train/checkpoint_2023-06-07_115922_2/AvgHist-AvgHist.pth"
output_path="$HOME/blob/output/ES_final_dataset/Avghist/train/checkpoint_2023-06-07_115922_2/"
valid_item_file="$HOME/data/ES_data/valid_merged_items.txt"
test_item_file="$HOME/data/ES_data/test_merged_items.txt"

# features_filepath="$HOME/blob/final_data/unirec_data/DE_final_dataset/id2features_2.csv"
###############################################################################################


# default parameters for local run
MY_DIR=$LOCAL_ROOT


DATASET_NAME="ES_final_dataset"  #"x-engmt-1m" #"Beauty"   

cd $MY_DIR
export PYTHONPATH=$PWD

### valid user ###################################
# CUDA_VISIBLE_DEVICES='0,1' torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 unirec/main/reco_topk.py \
CUDA_VISIBLE_DEVICES=0 python unirec/main/reco_topk.py \
    --dataset_name="valid_userids.csv" \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --model_file=$model_file \
    --test_batch_size=1024 \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --output_path=$output_path"valid_merge_scores.csv" \
    --last_item=1 \
    --topk=100 \
    --item_file=$valid_item_file \
    # --features_filepath=$features_filepath

### test user ###################################
# CUDA_VISIBLE_DEVICES='0,1' torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 unirec/main/reco_topk.py \
CUDA_VISIBLE_DEVICES=0 python unirec/main/reco_topk.py \
    --dataset_name="test_phase2.csv" \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --model_file=$model_file \
    --test_batch_size=1024 \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --output_path=$output_path"test_merge_scores.csv" \
    --last_item=0 \
    --topk=100 \
    --item_file=$test_item_file \
    # --features_filepath=$features_filepath
    # --item_file='$HOME/data/FR_data/test_merged_items.txt'