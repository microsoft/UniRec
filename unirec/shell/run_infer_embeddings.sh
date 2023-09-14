#!/bin/bash
# infer embeddings for given users or items
# root
LOCAL_ROOT='/home/v-leiyuxuan/working_dir/UniRec'

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="/home/v-leiyuxuan/blob/final_data/unirec_data"

DATASET_NAME="ES_final_dataset"

cd $MY_DIR
export PYTHONPATH=$PWD

model_file="/home/v-leiyuxuan/working_dir/UniRec/output/ES_final_dataset/SASRec/train/checkpoint_2023-06-25_072144_30/SASRec-SASRec.pth"
output_path="/home/v-leiyuxuan/working_dir/UniRec/output/ES_final_dataset/SASRec/train/checkpoint_2023-06-25_072144_30/"
### infer valid user embedding ###################################
# CUDA_VISIBLE_DEVICES='0,1' torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 unirec/main/infer_embedding.py \
CUDA_VISIBLE_DEVICES=0 python unirec/main/infer_embedding.py \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --id_file_name="valid_userids.csv" \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --model_file=$model_file \
    --test_batch_size=1024 \
    --output_emb_file=$output_path"valid_user_embedding.txt" \
    --node_type="user" \
    --last_item=1

### infer all item embeddings ###################################
# CUDA_VISIBLE_DEVICES='0,1' torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 unirec/main/infer_embedding.py \
CUDA_VISIBLE_DEVICES=0 python unirec/main/infer_embedding.py \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --model_file=$model_file \
    --test_batch_size=40960 \
    --output_emb_file=$output_path"item_embedding.txt" \
    --node_type="item"