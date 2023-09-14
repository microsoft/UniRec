#!/bin/bash 

if [ $# -eq 0 ]
  then
    ## No arguments, which means local execution
    LOCAL_ROOT='/media/xreco/MSRA/jianxun/UniRec' 
    
    MY_DIR=$LOCAL_ROOT
    ALL_DATA_ROOT="$LOCAL_ROOT/data"
    OUTPUT_ROOT="$LOCAL_ROOT/output" 
    MODEL_NAME='EASE' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec, EASE, SAR, SLIM, MultiVAE]
    loss_type='softmax' # [bce, bpr, softmax]
    DATASET_NAME="xbox_app_store"
    max_seq_len=20
    verbose=2

else
    ### execute on ITP
    LOCAL_ROOT='/home/jialia/UniRec'
    MY_DIR=$1 #"$LOCAL_ROOT/unirec"
    ALL_DATA_ROOT=$2 #"$LOCAL_ROOT/data"
    OUTPUT_ROOT=$3 #"$LOCAL_ROOT/output"
    MODEL_NAME=$4 # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec]
    loss_type=$5 #'softmax' # [bce, bpr, softmax] 
    max_seq_len=$6
    DATASET_NAME="xbox_app_store"
    verbose=0
fi

cd $MY_DIR
export PYTHONPATH=$PWD 

# overall config
DATA_TYPE='AERecDataset'  # BaseDataset SeqRecDataset AERecDataset

# train
learning_rate=0.002 
test_protocol='one_vs_all'  #'one_vs_all' 'session_aware'


# for  MODEL_NAME in 'AvgHist' 'AttHist' 'MF' 'SVDPlusPlus' 'GRU4Rec' 'SASRec'
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
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=1 \
    --has_item_bias=1 \
    --epochs=50  \
    --early_stop=10 \
    --batch_size=512 \
    --n_sample_neg_train=9 \
    --n_sample_neg_valid=9 \
    --test_protocol=$test_protocol \
    --grad_clip_value=0.1 \
    --weight_decay=1e-6 \
    --history_mask_mode='autoregressive' \
    --user_history_filename="user_history" \
    --metrics="['group_auc', 'hit@5;10;20', 'ndcg@5;10;20']" \
    --key_metric="ndcg@5" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose\

# done
# done

    