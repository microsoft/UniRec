# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash 

HOME_DIR=$(eval echo ~)
LOCAL_ROOT='$HOME_DIR/Unirec' 

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output" 
MODEL_NAME='ConvFormer' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU, SASRec, ConvFormer]
loss_type='bpr' # [bce, bpr, softmax]
DATASET_NAME="Beauty"
max_seq_len=50
verbose=2


cd $MY_DIR
export PYTHONPATH=$PWD 

# overall config
DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset

# train
learning_rate=0.002 
test_protocol='one_vs_all'  # 'one_vs_k' 'one_vs_all' 'session_aware'
history_mask_mode='autoregressive' # 'autoregressive'


# for  MODEL_NAME in 'AvgHist' 'AttHist' 'MF' 'SVDPlusPlus' 'GRU' 'SASRec'
# do
# for loss_type in 'bce' 'bpr' 'softmax' 
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
    --dropout_prob=0.0 \
    --embedding_size=32 \
    --hidden_size=32 \
    --inner_size=32 \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=1 \
    --has_item_bias=1 \
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
    --user_history_filename="user_history" \
    --metrics="['hit@10;20;30', 'ndcg@10;20;30']" \
    --key_metric="ndcg@20" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --neg_by_pop_alpha=0 \
# done
# done

    
