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
MODEL_NAME='SASRec' # [ MF, AvgHist, AttHist, SVDPlusPlus, GRU, SASRec, ConvFormer, FASTConvFormer]
loss_type='bce' # [bce, bpr, softmax]
DATASET_NAME="ml-100k"
max_seq_len=10
verbose=2

cd $MY_DIR
export PYTHONPATH=$PWD 

# overall config
DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset

# train
learning_rate=0.001
test_protocol='one_vs_all'  # 'one_vs_k' 'one_vs_all' 'session_aware'


exp_name="$MODEL_NAME-$DATASET_NAME"

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
    --embedding_size=64 \
    --hidden_size=64 \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=100  \
    --early_stop=10 \
    --batch_size=512 \
    --n_sample_neg_train=5 \
    --neg_by_pop_alpha=1.0 \
    --valid_protocol=$test_protocol \
    --test_protocol=$test_protocol \
    --grad_clip_value=-1 \
    --weight_decay=1e-6 \
    --history_mask_mode='autoregressive' \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --metrics="['hit@10;20', 'ndcg@10;20']" \
    --key_metric="ndcg@10" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --exp_name=$exp_name \
    --use_wandb=0
# done