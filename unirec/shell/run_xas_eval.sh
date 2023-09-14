#!/bin/bash 

# root
LOCAL_ROOT='/media/xreco/MSRA/jianxun/UniRec'

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"

cd $MY_DIR
export PYTHONPATH=$PWD

DATASET_NAME="xbox_app_store" 

# overall config
# MODEL_NAME='AttHist' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec]
# model_file="/media/xreco/MSRA/jianxun/UniRec/output/AvgHist_xbox_app_store/train/checkpoint_2022-11-01_132022_63/AvgHist-main.pth"
# DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset
 

# ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
# mkdir -p $ALL_RESULTS_ROOT
# ### train ###################################
# python main/main.py \
#     --config_dir="config" \
#     --model=$MODEL_NAME \
#     --dataloader=$DATA_TYPE \
#     --dataset=$DATASET_NAME \
#     --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
#     --output_path=$ALL_RESULTS_ROOT"/eval" \
#     --batch_size=512 \
#     --test_protocol='session_aware' \
#     --valid_protocol='one_vs_k' \
#     --history_mask_mode='autoregressive' \
#     --metrics="['group_auc', 'hit@5;10;20', 'ndcg@5;10;20']" \
#     --num_workers=4 \
#     --verbose=2 \
#     --task='infer' \
#     --model_file=$model_file


# # overall config
# MODEL_NAME='AvgHist' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec]
# model_file="/media/xreco/MSRA/jianxun/UniRec/output/AvgHist_xbox_app_store/train/checkpoint_2022-11-01_132022_63/AvgHist-main.pth"
# DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset
 
# ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
# mkdir -p $ALL_RESULTS_ROOT
# ### train ###################################
# python main/main.py \
#     --config_dir="config" \
#     --model=$MODEL_NAME \
#     --dataloader=$DATA_TYPE \
#     --dataset=$DATASET_NAME \
#     --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
#     --data_test_name="purchase_test_full_candidate" \
#     --test_file_format="user-item" \
#     --output_path=$ALL_RESULTS_ROOT"/eval" \
#     --batch_size=512 \
#     --test_protocol='session_aware' \
#     --user_history_filename="user_history" \
#     --valid_protocol='one_vs_k' \
#     --history_mask_mode='autoregressive' \
#     --metrics="['group_auc', 'hit@5;10;20', 'ndcg@5;10;20']" \
#     --num_workers=0 \
#     --verbose=2 \
#     --task='infer' \
#     --model_file=$model_file



### for the app_store_new dataset
DATASET_NAME="xas_1114" 
MODEL_NAME='AvgHist' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec]
model_file="/media/xreco/MSRA/jianxun/UniRec/output/AvgHist_xas_1114/train/checkpoint_2022-11-14_153242_5/AvgHist-main.pth"
DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset
task='infer' # 'infer' 'test'
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT
### train ###################################
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --data_test_name="engmt_test_full_candidate" \
    --test_file_format="user-item" \
    --output_path=$ALL_RESULTS_ROOT"/eval" \
    --batch_size=512 \
    --test_protocol='session_aware' \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq-time_seq" \
    --history_mask_mode='unorder' \
    --metrics="['group_auc', 'hit@5;10;20', 'ndcg@5;10;20']" \
    --num_workers_test=0 \
    --verbose=2 \
    --task=$task \
    --model_file=$model_file

# # SAR
# DATASET_NAME="xbox_app_store_new" 
# MODEL_NAME='SAR'  
# DATA_TYPE='BaseDataset'  # BaseDataset SeqRecDataset  
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
#     --user_history_file_format="user-item_seq-time_seq" \
#     --metrics="['group_auc', 'hit@5;10;20', 'ndcg@5;10;20']" \
#     --key_metric="ndcg@5" \
#     --verbose=2 \
#     --task='test'

        