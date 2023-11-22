# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
RAW_DATA_DIR="/path/to/raw_datasets/CF_8_1_1"

ROOT_DIR='/path/to/UniRec'
###############################################################################################


# default parameters for local run
MY_DIR=$ROOT_DIR
DATA_ROOT="$ROOT_DIR/data"
OUTPUT_ROOT="$ROOT_DIR/output"


dataset_name='yelp2018' ## gowalla amazon-book yelp2018

export PYTHONPATH=$MY_DIR

raw_datapath="$RAW_DATA_DIR/$dataset_name" 
dataset_outpathroot=$DATA_ROOT
example_yaml_file="$MY_DIR/unirec/config/dataset/example.yaml"
 

cd $MY_DIR"/preprocess"
python prepare_data.py \
    --raw_datapath=$raw_datapath \
    --outpathroot=$dataset_outpathroot \
    --dataset_name=$dataset_name \
    --example_yaml_file=$example_yaml_file \
    --index_by_zero=1 \
    --sep=" "  \
    --train_file='train.txt'\
    --train_file_format='user_item_seq' \
    --train_file_col_names="['user_id', 'item_seq']" \
    --train_file_has_header=0 \
    --train_neg_k=0 \
    --valid_file='val.txt'\
    --valid_file_format='user_item_seq' \
    --valid_file_col_names="['user_id', 'item_seq']" \
    --valid_file_has_header=0 \
    --valid_neg_k=0 \
    --test_file='test.txt'\
    --test_file_format='user_item_seq' \
    --test_file_col_names="['user_id', 'item_seq']" \
    --test_file_has_header=0 \
    --test_neg_k=0 \
    