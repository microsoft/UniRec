#!/bin/bash

LOCAL_ROOT="/home/jialia/adaretriever/UniRec"

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"



# default parameters for local run
MODEL_NAME='SASRec' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec]
DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset
DATASET_NAME="Beauty" #"Steam"   
verbose=2
learning_rate=0.001
epochs=200
weight_decay=0 #1e-6
dropout_prob=0.1777
loss_type='bce' # [bce, bpr, softmax, ccl]
ccl_w=100
ccl_m=0.4
distance_type='dot' # [cosine, mlp, dot]
n_sample_neg_train=100  #400
max_seq_len=50
history_mask_mode='autoregressive'
embedding_size=64

## for ITP executation, we pass through arguments
if [ $# -gt 0 ]
then
    ### execute on ITP
    MY_DIR=$1 #"$LOCAL_ROOT/unirec"
    ALL_DATA_ROOT=$2 #"$LOCAL_ROOT/data"
    OUTPUT_ROOT=$3 #"$LOCAL_ROOT/output"
    MODEL_NAME=$4 # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec]
    loss_type=$5 #'softmax' # [bce, bpr, softmax] 
    max_seq_len=$6
    DATASET_NAME=$7
    history_mask_mode=$8
    learning_rate=$9
    verbose=0
fi

cd $MY_DIR
export PYTHONPATH=$PWD


# for ccl_w in 10 100 400
# do
# for embedding_size in 64 256
# do
# for dropout_prob in  0 0.2 0.5
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
    --dropout_prob=$dropout_prob \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=0 \
    --has_item_bias=1 \
    --epochs=$epochs  \
    --batch_size=1024 \
    --n_sample_neg_train=$n_sample_neg_train \
    --n_sample_neg_valid=99 \
    --valid_protocol='one_vs_all' \
    --test_protocol='one_vs_all' \
    --grad_clip_value=10 \
    --weight_decay=$weight_decay \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq"  \
    --history_mask_mode=$history_mask_mode \
    --metrics="['hit@5;10;20', 'ndcg@5;10;20']" \
    --key_metric="ndcg@10" \
    --shuffle_train=1 \
    --seed=366849 \
    --early_stop=10 \
    --embedding_size=$embedding_size \
    --hidden_size=$embedding_size \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --neg_by_pop_alpha=0 \
    --distance_type=$distance_type \
    --ccl_w=$ccl_w \
    --ccl_m=$ccl_m \
    --hidden_dropout_prob=$dropout_prob \
    --attn_dropout_prob=$dropout_prob \
    --n_layers=2 \
    --n_heads=4 \
    --use_wandb=0 \
    --wandb_file="$LOCAL_ROOT/unirec/shell/ada-retrieval/wandb.yaml" \
    # --gpu_id=0 #if you want to use parallelize on a multi-GPU machine, you should set gpu_id=-1 and use CUDA_VISIBLE_DEVICES to specify the GPU ids to use. See run_wandb_hypertune.sh for more details.

# done
# done
# # done


# MODEL_NAME='SAR'   # SAR, EASE, SLIM, AdmmSLIM
# DATA_TYPE='AERecDataset'  # BaseDataset SeqRecDataset
# test_protocol='one_vs_all'  #'one_vs_all' 'session_aware' 
# user_history_filename='train' #'user_history'
# user_history_file_format='user-item' #"user-item_seq" 
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
#     --user_history_filename=$user_history_filename \
#     --user_history_file_format=$user_history_file_format \
#     --metrics="['hit@5;10;20', 'ndcg@5;10;20']" \
#     --key_metric="hit@10" \
#     --verbose=2 \