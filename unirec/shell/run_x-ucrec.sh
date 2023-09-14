#!/bin/bash

## modified on 2023/04/05

# root
LOCAL_ROOT='/home/jialia/UniRec'

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"



# default parameters for local run
MODEL_NAME='AvgHist' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec]
DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset
DATASET_NAME="x-ucrec-1m" #"Beauty"   
verbose=2
learning_rate=0.001
epochs=50
weight_decay=0 #1e-6
dropout_prob=0
loss_type='bce' # [bce, bpr, softmax, ccl]
ccl_w=100
ccl_m=0.4
distance_type='dot' # [cosine, mlp, dot]
n_sample_neg_train=19  #400
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
 
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT

task="reco_topk"  ## "embedding"  "train"  "test" "infer" "reco_topk"
## check if the value of $task is train or test
if [[ $task == "train" ]]; then
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
    --has_item_bias=0 \
    --epochs=$epochs  \
    --batch_size=512 \
    --n_sample_neg_train=$n_sample_neg_train \
    --n_sample_neg_valid=99 \
    --valid_protocol='one_vs_all' \
    --test_protocol='one_vs_all' \
    --grad_clip_value=10 \
    --weight_decay=$weight_decay \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item"  \
    --history_mask_mode=$history_mask_mode \
    --metrics="['hit@5;10;20', 'ndcg@5;10;20']" \
    --key_metric="ndcg@10" \
    --shuffle_train=1 \
    --seed=2022 \
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
    --attn_dropout_prob=$dropout_prob
elif [[ $task == "test" || $task == "infer" ]]; then
### test ################################### 
model_file=/home/jialia/UniRec/output/x-ucrec-1m/AvgHist/train/checkpoint_2023-04-05_043538_9/AvgHist-AvgHist.pth
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --data_test_name="test" \
    --test_file_format="user-item" \
    --output_path=$ALL_RESULTS_ROOT"/eval" \
    --batch_size=512 \
    --test_protocol='one_vs_all' \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item" \
    --history_mask_mode=$history_mask_mode \
    --metrics="['group_auc', 'hit@5;10;20', 'ndcg@5;10;20']" \
    --num_workers_test=0 \
    --verbose=2 \
    --task=$task \
    --model_file=$model_file
elif [[ $task == "embedding" ]]; then
### infer embedding ################################### 
model_file=/home/jialia/UniRec/output/x-ucrec-1m/AvgHist/train/checkpoint_2023-04-05_043538_9/AvgHist-AvgHist.pth
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --data_test_name="all_user4emb" \
    --test_file_format="user-item" \
    --output_path=$ALL_RESULTS_ROOT"/embbedding" \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item" \
    --history_mask_mode=$history_mask_mode \
    --num_workers_test=0 \
    --verbose=2 \
    --task=$task \
    --model_file=$model_file
elif [[ $task == "reco_topk" ]]; then
model_file=/home/jialia/UniRec/output/x-ucrec-1m/AvgHist/train/checkpoint_2023-04-05_043538_9/AvgHist-AvgHist.pth
python unirec/main/reco_topk.py \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --dataset_name="test_userids.txt" \
    --output_path=$ALL_RESULTS_ROOT"/topk" \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item" \
    --model_file=$model_file \
    --test_batch_size=10
fi