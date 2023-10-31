# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash 
HOME_DIR=$(eval echo ~)
LOCAL_ROOT='$HOME_DIR/work/UniRec' 

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output" 
MODEL_NAME='MF' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU, SASRec, ConvFormer, FASTConvFormer]
loss_type='bpr' # [bce, bpr, softmax]
DATASET_NAME="amazon-electronics"
max_seq_len=50
verbose=2

cd $MY_DIR
export PYTHONPATH=$PWD 

# overall config
DATA_TYPE='BaseDataset'  # BaseDataset SeqRecDataset

# train
learning_rate=0.001
test_protocol='one_vs_all'  # 'one_vs_k' 'one_vs_all' 'session_aware'

# required arguments
enable_morec=1
ngroup=10
alpha=0.01
lambda=0.2
expect_loss=0.25
beta_min=0.1
beta_max=1.5
K_p=0.05
K_i=0.001
objective_weights="[0.1,0.1,0.8]"   # weight for objectives expect accuracy

item_meta_morec_filename="item_meta_morec_filename.tsv"
alignment_distribution_filename="align_dist.tsv"

# well-trained model
model_file="$HOME_DIR/work/UniRec/output/amazon-electronics/MF/train/amazon-electronics_MF_bpr/MF-MF.pth"

currentTime=`date "+%Y-%m-%d_%H%M%S"`
exp_name="MoRec-FinetuneModel"
checkpoint_dir=$DATASET_NAME"_"$MODEL_NAME"_"$loss_type"_MoRecFinetune_"$currentTime

# for  MODEL_NAME in 'AvgHist' 'AttHist' 'MF' 'SVDPlusPlus' 'GRU' 'SASRec'
# do
# for loss_type in 'bce' 'bpr' 'softmax' 
# do
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT
### train ###################################

# python -m debugpy --listen 9999 --wait-for-client main/main.py \
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --epochs=30  \
    --early_stop=-1 \
    --metrics="['hit@10', 'ndcg@10', 'rhit@10', 'rndcg@10', 'pop-kl@10', 'least-misery']" \
    --key_metric="ndcg@10" \
    --enable_morec=$enable_morec \
    --morec_objectives 'fairness' 'alignment' 'revenue' \
    --morec_ngroup=$ngroup \
    --morec_alpha=$alpha \
    --morec_lambda=$lambda \
    --morec_expect_loss=$expect_loss \
    --morec_beta_min=$beta_min \
    --morec_beta_max=$beta_max \
    --morec_K_p=$K_p \
    --morec_K_i=$K_i \
    --morec_objective_weights=$objective_weights \
    --item_meta_morec_filename=$item_meta_morec_filename \
    --align_dist_filename=$alignment_distribution_filename \
    --model_file=$model_file \
    --checkpoint_dir=$checkpoint_dir \
    --exp_name=$exp_name \
    --use_tensorboard=1 \
# done
# done

    
