#!/bin/bash

# root
LOCAL_ROOT='/media/xreco/MSRA/yuxuan/UniRec'

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"



# default parameters for local run
MODEL_NAME='LKNN' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec, LKNN, MultiVAE]
DATA_TYPE='SeqRecDataset' #AERecDataset BaseDataset SeqRecDataset
DATASET_NAME="x-benchmark-seq-1m"  #"x-engmt-1m" #"Beauty"   
verbose=2
learning_rate=0.0001
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
embedding_size=32

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
    --has_user_bias=1 \
    --has_item_bias=1 \
    --epochs=$epochs  \
    --batch_size=512 \
    --n_sample_neg_train=$n_sample_neg_train \
    --n_sample_neg_valid=99 \
    --valid_protocol='one_vs_all' \
    --test_protocol='one_vs_all' \
    --grad_clip_value=10 \
    --weight_decay=$weight_decay \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq-time_seq"  \
    --history_mask_mode=$history_mask_mode \
    --metrics="['hit@5;10;20', 'ndcg@5;10;20']" \
    --key_metric="ndcg@10" \
    --shuffle_train=0 \
    --seed=2022 \
    --early_stop=5 \
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
    --time_seq=101

# done
# done
# # done