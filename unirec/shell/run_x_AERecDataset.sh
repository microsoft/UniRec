#!/bin/bash 

# root
LOCAL_ROOT='/media/xreco/MSRA/jianxun/UniRec'

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"

cd $MY_DIR
export PYTHONPATH=$PWD

# DATASET_NAME="xbox_app_store_small" 
# MODEL_NAME='SAR'   # SAR, EASE, SLIM, AdmmSLIM
# DATA_TYPE='AERecDataset'  # BaseDataset SeqRecDataset
# test_protocol='one_vs_all'  #'one_vs_all' 'session_aware' 
# ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
# mkdir -p $ALL_RESULTS_ROOT 
# ### do train and evaluation on test test
# python main/main.py \
#     --config_dir="config" \
#     --model=$MODEL_NAME \
#     --dataloader=$DATA_TYPE \
#     --dataset=$DATASET_NAME \
#     --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
#     --output_path=$ALL_RESULTS_ROOT"/train" \
#     --test_protocol=$test_protocol \
#     --history_mask_mode='unorder' \
#     --user_history_filename="user_history" \
#     --user_history_file_format="user-item_seq-time_seq" \
#     --metrics="['group_auc', 'hit@5;10;20', 'ndcg@5;10;20']" \
#     --key_metric="ndcg@5" \
#     --verbose=2 \


# DATASET_NAME="x-engmt-1m"
# MODEL_NAME='SAR'   # SAR, EASE, SLIM, AdmmSLIM
# DATA_TYPE='AERecDataset'  # BaseDataset SeqRecDataset
# test_protocol='one_vs_all'  #'one_vs_all' 'session_aware' 
# ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
# mkdir -p $ALL_RESULTS_ROOT 
# python main/main.py \
#     --config_dir="config" \
#     --model=$MODEL_NAME \
#     --dataloader=$DATA_TYPE \
#     --dataset=$DATASET_NAME \
#     --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
#     --output_path=$ALL_RESULTS_ROOT"/train" \
#     --test_protocol=$test_protocol \
#     --history_mask_mode='unorder' \
#     --user_history_filename="user_history" \
#     --user_history_file_format="user-item_seq" \
#     --metrics="['hit@10;20;30', 'ndcg@10;20;30']" \
#     --key_metric="ndcg@20" \
#     --verbose=2 \

DATASET_NAME="x-engmt-1m"
MODEL_NAME='SAR'   # SAR, EASE, SLIM, AdmmSLIM
DATA_TYPE='AERecDataset'  # BaseDataset SeqRecDataset
test_protocol='one_vs_all'  #'one_vs_all' 'session_aware' 
user_history_filename='train' #'user_history'
user_history_file_format='user-item' #"user-item_seq" 
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT 
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --test_protocol=$test_protocol \
    --history_mask_mode='unorder' \
    --user_history_filename=$user_history_filename \
    --user_history_file_format=$user_history_file_format \
    --metrics="['hit@10;20;30', 'ndcg@10;20;30']" \
    --key_metric="ndcg@20" \
    --verbose=2 \

DATASET_NAME="x-engmt-1m"
MODEL_NAME='MultiVAE'
DATA_TYPE='AERecDataset'  # BaseDataset SeqRecDataset
test_protocol='one_vs_all'  #'one_vs_all' 'session_aware'
user_history_filename='train' #'user_history'
user_history_file_format='user-item' #"user-item_seq" 
learning_rate=0.002 
history_mask_mode='autoregressive' # 'autoregressive'
loss_type='bpr' 
max_seq_len=50
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT 
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --learning_rate=$learning_rate \
    --dropout_prob=0.0 \
    --embedding_size=32 \
    --hidden_size=32 \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=50  \
    --early_stop=5 \
    --batch_size=512 \
    --n_sample_neg_train=9 \
    --n_sample_neg_valid=19 \
    --valid_protocol=$test_protocol \
    --test_protocol=$test_protocol \
    --grad_clip_value=0.1 \
    --weight_decay=1e-6 \
    --history_mask_mode=$history_mask_mode \
    --user_history_filename=$user_history_filename \
    --user_history_file_format=$user_history_file_format \
    --metrics="['hit@10;20;30', 'ndcg@10;20;30']" \
    --key_metric="ndcg@20" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=2

# ### do train and score inference on test set, no evaluation with labels
# python main/main.py \
#     --config_dir="config" \
#     --model=$MODEL_NAME \
#     --dataloader=$DATA_TYPE \
#     --dataset=$DATASET_NAME \
#     --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
#     --output_path=$ALL_RESULTS_ROOT"/train" \
#     --test_protocol=$test_protocol \
#     --history_mask_mode='unorder' \
#     --user_history_filename="user_history" \
#     --user_history_file_format="user-item_seq-time_seq" \
#     --metrics="['group_auc', 'hit@5;10;20', 'ndcg@5;10;20']" \
#     --key_metric="ndcg@5" \
#     --verbose=2 \
#     --task='infer' \
#     --test_protocol='session_aware' \
#     --num_workers=0 \
 