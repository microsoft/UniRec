# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
ROOT_DIR="$HOME/workspace/UniRec"  # path to UniRec
###############################################################################################


# default parameters for local run
RAW_DATA_DIR="$HOME/.unirec/dataset"
RAW_DATA_PREFILE="$RAW_DATA_DIR/ml-100k/full_user_history.csv"
RAW_DATA_FILE="$RAW_DATA_DIR/ml-100k-rank/ml-100k-rank.txt"

MY_DIR=$ROOT_DIR
DATA_ROOT="$ROOT_DIR/data"
OUTPUT_ROOT="$ROOT_DIR/output"


dataset_name='ml-100k-rank' ## gowalla amazon-book yelp2018

export PYTHONPATH=$MY_DIR

raw_datapath="$RAW_DATA_DIR/$dataset_name" 
dataset_outpathroot=$DATA_ROOT
example_yaml_file="$MY_DIR/unirec/config/dataset/example.yaml"

# group_size=21
# n_neg_k=$((group_size-1)) # if use group_size

group_size=-1
n_neg_k=20
pretrain_word2vec=1
embedding_size=32


cd $MY_DIR"/examples/preprocess"
# run ranker.py to get rank data in T4 data format from user history
python specific_datasets/ranker.py \
    --data_format='rank' \
    --infile=$RAW_DATA_FILE \
    --outdir=$raw_datapath \
    --n_neg_k=$n_neg_k \
    --prefile=$RAW_DATA_PREFILE \
    --prefile_file_format='user-item_seq' \
    --sep="\t" \
    --pretrain_word2vec=$pretrain_word2vec \
    --embedding_size=$embedding_size


python prepare_data.py \
    --raw_datapath=$raw_datapath \
    --outpathroot=$dataset_outpathroot \
    --dataset_name=$dataset_name \
    --example_yaml_file=$example_yaml_file \
    --index_by_zero=0 \
    --sep=" " \
    --train_file='train.txt' \
    --train_file_format='user-item_group-label_group' \
    --train_file_col_names="['user_id', 'item_id_list', 'label_list']" \
    --train_file_has_header=0 \
    --train_neg_k=0 \
    --valid_file='valid.txt' \
    --valid_file_format='user-item_group-label_group' \
    --valid_file_col_names="['user_id', 'item_id_list', 'label_list']" \
    --valid_file_has_header=0 \
    --valid_neg_k=0 \
    --test_file='test.txt' \
    --test_file_format='user-item_group-label_group' \
    --test_file_col_names="['user_id', 'item_id_list', 'label_list']" \
    --test_file_has_header=0 \
    --test_neg_k=0 \
    --user_history_file='user_history.txt' \
    --user_history_file_format='user-item_seq' \
    --user_history_file_has_header=0 \
    --user_history_file_col_names="['user_id', 'item_seq']" \
    --group_size=$group_size
