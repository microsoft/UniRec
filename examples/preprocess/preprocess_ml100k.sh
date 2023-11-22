# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
HOME_DIR=$(eval echo ~)
ROOT_DIR="$HOME_DIR/work/UniRec"    # Your path to UniRec


###############################################################################################
############################## default parameters for local run ###############################
###############################################################################################
RAW_DATA_DIR="$HOME_DIR/.unirec/dataset/ml-100k"

MY_DIR=$ROOT_DIR
DATA_ROOT="$ROOT_DIR/data"
OUTPUT_ROOT="$ROOT_DIR/output"

# dataset_name='ml-25m-retrieval'
dataset_name='ml-100k'


export PYTHONPATH=$MY_DIR

raw_datapath="$RAW_DATA_DIR" 
dataset_outpathroot=$DATA_ROOT
example_yaml_file="$MY_DIR/unirec/config/dataset/example.yaml"
 

cd $MY_DIR"/examples/preprocess"
python prepare_data.py \
    --raw_datapath=$raw_datapath \
    --outpathroot=$dataset_outpathroot \
    --dataset_name=$dataset_name \
    --example_yaml_file=$example_yaml_file \
    --index_by_zero=0 \
    --sep="\t"  \
    --train_file='train.csv'\
    --train_file_format='user-item' \
    --train_file_has_header=1 \
    --train_file_col_names="['user_id', 'item_id']" \
    --train_neg_k=0 \
    --valid_file='valid.csv'\
    --valid_file_format='user-item' \
    --valid_file_has_header=1 \
    --valid_file_col_names="['user_id', 'item_id']" \
    --valid_neg_k=0 \
    --test_file='test.csv'\
    --test_file_format='user-item' \
    --test_file_has_header=1 \
    --test_file_col_names="['user_id', 'item_id']" \
    --test_neg_k=0 \
    --user_history_file='user_history.csv'\
    --user_history_file_format='user-item_seq' \
    --user_history_file_has_header=1 \
    --user_history_file_col_names="['user_id', 'item_seq']"
