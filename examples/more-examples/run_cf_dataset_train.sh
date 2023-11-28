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
MODEL_NAME='MF' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU, SASRec]
loss_type='bpr' # [bce, bpr, softmax]
DATASET_NAME="amazon-book" # gowalla amazon-book yelp2018
max_seq_len=20
verbose=2
history_mask_mode='unorder' # 'autoregressive'
learning_rate=0.002


cd $MY_DIR
export PYTHONPATH=$PWD 

# overall config
DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset


# # for  MODEL_NAME in 'AvgHist' 'AttHist' 'MF' 'SVDPlusPlus' 'GRU' 'SASRec'
# # do
# # for loss_type in 'bce' 'bpr' 'softmax' 
# # do
# ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
# mkdir -p $ALL_RESULTS_ROOT
# ### train ###################################
# python main/main.py \
#     --config_dir="config" \
#     --model=$MODEL_NAME \
#     --dataloader=$DATA_TYPE \
#     --dataset=$DATASET_NAME \
#     --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
#     --output_path=$ALL_RESULTS_ROOT"/train" \
#     --learning_rate=$learning_rate \
#     --dropout_prob=0.0 \
#     --embedding_size=32 \
#     --hidden_size=32 \
#     --use_pre_item_emb=0 \
#     --loss_type=$loss_type \
#     --max_seq_len=$max_seq_len \
#     --has_user_bias=1 \
#     --has_item_bias=1 \
#     --epochs=100  \
#     --early_stop=10 \
#     --batch_size=512 \
#     --n_sample_neg_train=9 \
#     --n_sample_neg_valid=0 \
#     --valid_protocol="one_vs_all" \
#     --test_protocol="one_vs_all" \
#     --grad_clip_value=0.1 \
#     --weight_decay=1e-6 \
#     --history_mask_mode=$history_mask_mode \
#     --user_history_filename="train" \
#     --metrics="['hit@10;20;50', 'ndcg@10;20;50', 'recall@10;20;50']" \
#     --key_metric="ndcg@20" \
#     --num_workers=4 \
#     --num_workers_test=0 \
#     --num_workers_valid=0 \
#     --verbose=$verbose \
#     --neg_by_pop_alpha=0
# # done
# # done

MODEL_NAME='SAR'   # SAR, EASE, SLIM, AdmmSLIM
DATA_TYPE='AERecDataset'  # BaseDataset SeqRecDataset
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT  
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --n_sample_neg_train=9 \
    --n_sample_neg_valid=0 \
    --valid_protocol="one_vs_all" \
    --test_protocol="one_vs_all" \
    --history_mask_mode='unorder' \
    --user_history_filename="train" \
    --metrics="['hit@10;20;50', 'ndcg@10;20;50', 'recall@10;20;50']" \
    --key_metric="ndcg@20" \
    --verbose=2 \