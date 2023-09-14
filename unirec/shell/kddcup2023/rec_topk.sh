#!/bin/bash
# get the top k items for every user in valid set and test set,  and save the top k items in a file
# root
LOCAL_ROOT='/home/v-leiyuxuan/working_dir/UniRec'

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="/home/v-leiyuxuan/blob/final_data/unirec_data"

# default parameters for local run
DATASET_NAME="ES_final_dataset"  #"x-engmt-1m" #"Beauty"   

cd $MY_DIR
export PYTHONPATH=$PWD

model_file="/home/v-leiyuxuan/working_dir/UniRec/output/ES_final_next_item_dataset/SASRec/train/checkpoint_2023-06-24_035204_1/SASRec-SASRec.pth"
output_path="/home/v-leiyuxuan/working_dir/UniRec/output/ES_final_next_item_dataset/SASRec/train/checkpoint_2023-06-24_035204_1/"
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
# --features_filepath="/home/v-leiyuxuan/blob/final_data/unirec_data/JP_final_dataset/id2features_2.csv"
# --item_file='/home/v-leiyuxuan/yuxuan_lei/data/FR_data/test_merged_items.txt'