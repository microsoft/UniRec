# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

RAW_DATA_DIR="/path/to/raw_datasets"

ROOT_DIR='/path/to/UniRec'
MY_DIR=$ROOT_DIR
DATA_ROOT="$ROOT_DIR/data"
OUTPUT_ROOT="$ROOT_DIR/output"

# dataset_name='ml-25m-retrieval'
dataset_name='ml-25m-retrieval_firstlines'


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
    --index_by_zero=0 \
    --sep=","  \
    --train_file='train.csv'\
    --train_file_format='user-item' \
    --train_file_has_header=1 \
    --train_file_col_names="['user_id', 'item_id', 'rating', 'timestamp']" \
    --train_neg_k=0 \
    --valid_file='valid.csv'\
    --valid_file_format='user-item' \
    --valid_file_has_header=1 \
    --valid_file_col_names="['user_id', 'item_id', 'rating', 'timestamp']" \
    --valid_neg_k=0 \
    --test_file='test.csv'\
    --test_file_format='user-item' \
    --test_file_has_header=1 \
    --test_file_col_names="['user_id', 'item_id', 'rating', 'timestamp']" \
    --test_neg_k=0 \
