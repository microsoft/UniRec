# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash
# get the top k items for every user in valid set and test set,  and save the top k items in a file

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
HOME_DIR=$(eval echo ~)
LOCAL_ROOT='$HOME_DIR/UniRec'

ALL_DATA_ROOT="$HOME_DIR/blob/final_data/unirec_data"

model_file="$HOME_DIR/working_dir/UniRec/output/ES_final_next_item_dataset/SASRec/train/checkpoint_2023-06-24_035204_1/SASRec-SASRec.pth"
output_path="$HOME_DIR/working_dir/UniRec/output/ES_final_next_item_dataset/SASRec/train/checkpoint_2023-06-24_035204_1/"


###############################################################################################
############################## default parameters for local run ###############################
###############################################################################################
MY_DIR=$LOCAL_ROOT


DATASET_NAME="ES_final_dataset" 

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
    --output_path=$output_path"valid_top100.txt" \
    --last_item=1 \
    --topk=100

### test user ###################################
# CUDA_VISIBLE_DEVICES='0,1' torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 unirec/main/reco_topk.py \
CUDA_VISIBLE_DEVICES=0 python unirec/main/reco_topk.py \
    --dataset_name="test_phase2.csv" \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --model_file=$model_file \
    --test_batch_size=1024 \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --output_path=$output_path"test_top100.txt" \
    --last_item=0 \
    --topk=100
# --features_filepath="$HOME_DIR/blob/final_data/unirec_data/JP_final_dataset/id2features_2.csv"
# --item_file='$HOME_DIR/data/FR_data/test_merged_items.txt'