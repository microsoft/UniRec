# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash
# pre-train on one locale dataset with feature embedding and text embedding

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
HOME_DIR=$(eval echo ~)
LOCAL_ROOT='$HOME_DIR/working_dir/UniRec'

ALL_DATA_ROOT="$HOME_DIR/blob/final_data/unirec_data"

text_emb_path='$HOME_DIR/blob/final_data/unirec_data/ES_final_dataset/text_embs/addnextitem_mean_mbart/item_embeddings_nid.csv'
features_filepath='$HOME_DIR/blob/final_data/unirec_data/ES_final_dataset/id2features_2.csv'
wandb_file="$LOCAL_ROOT/unirec/shell/kddcup2023/wandb.yaml"


###############################################################################################
############################## default parameters for local run ###############################
###############################################################################################
MY_DIR=$LOCAL_ROOT
OUTPUT_ROOT="$LOCAL_ROOT/output"


MODEL_NAME='SASRec' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU, SASRec, ConvFormer, MultiVAE]
DATA_TYPE='SeqRecDataset' #AERecDataset BaseDataset SeqRecDataset
DATASET_NAME="ES_final_dataset"
verbose=2
learning_rate=0.00027532020029371717
epochs=100
weight_decay=0 #1e-6
dropout_prob=0
loss_type='fullsoftmax' # [bce, bpr, softmax, ccl, fullsoftmax]
distance_type='dot' # [cosine, mlp, dot]
n_sample_neg_train=0  #400
max_seq_len=7
history_mask_mode='autoregressive'
embedding_size=176

cd $MY_DIR
export PYTHONPATH=$PWD


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
    --batch_size=2048 \
    --n_sample_neg_train=$n_sample_neg_train \
    --n_sample_neg_valid=0 \
    --valid_protocol='one_vs_all' \
    --test_protocol='one_vs_all' \
    --grad_clip_value=23.436523771594445 \
    --weight_decay=$weight_decay \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq"  \
    --history_mask_mode=$history_mask_mode \
    --metrics="['hit@20;100;200;300;1000;2000;3000', 'ndcg@20;100;200;300;1000','mrr@20;100;200;300;1000']" \
    --key_metric="mrr@100" \
    --shuffle_train=1 \
    --seed=203737 \
    --early_stop=5 \
    --embedding_size=$embedding_size \
    --num_workers=6 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --neg_by_pop_alpha=0 \
    --distance_type=$distance_type \
    --hidden_dropout_prob=0.4305400601023631 \
    --attn_dropout_prob=0.1459099776835039 \
    --scheduler_factor=0.1209259552381572 \
    --tau=0.6952599498943698 \
    --use_text_emb=1 \
    --text_emb_path=$text_emb_path \
    --text_emb_size=1024 \
    --use_features=1 \
    --features_filepath=$features_filepath  \
    --features_shape='[3489, 99]' \
    --use_wandb=0 \
    --wandb_file=$wandb_file