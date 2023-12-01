# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
ROOT_DIR="$HOME/workspace/UniRec" # path to UniRec
###############################################################################################


# default parameters for local run
RAW_DATA_DIR="$HOME/.unirec/dataset/"

MY_DIR=$ROOT_DIR
DATA_ROOT="$ROOT_DIR/data"
OUTPUT_ROOT="$ROOT_DIR/output"

dataset_name='ml-100k-max_len'


export PYTHONPATH=$MY_DIR

raw_datapath="$RAW_DATA_DIR/$dataset_name" 
dataset_outpathroot=$DATA_ROOT
example_yaml_file="$MY_DIR/unirec/config/dataset/example.yaml"

DATA_TYPE='SeqRecDataset'

cd $MY_DIR"/examples/preprocess"
echo $PWD
python download_split_ml100k.py \
    --output_name=$dataset_name \
    --need_max_len=1 \

python prepare_data.py \
    --raw_datapath=$raw_datapath \
    --outpathroot=$dataset_outpathroot \
    --dataset_name=$dataset_name \
    --example_yaml_file=$example_yaml_file \
    --data_type=$DATA_TYPE \
    --index_by_zero=0 \
    --sep="\t"  \
    --train_file='train.csv'\
    --train_file_format='user-item-max_len' \
    --train_file_has_header=1 \
    --train_file_col_names="['user_id', 'item_id', 'max_len']" \
    --train_neg_k=0 \
    --valid_file='valid.csv'\
    --valid_file_format='user-item-max_len' \
    --valid_file_has_header=1 \
    --valid_file_col_names="['user_id', 'item_id', 'max_len']" \
    --valid_neg_k=0 \
    --test_file='test.csv'\
    --test_file_format='user-item-max_len' \
    --test_file_has_header=1 \
    --test_file_col_names="['user_id', 'item_id', 'max_len']" \
    --test_neg_k=0 \
    --user_history_file='user_history.csv'\
    --user_history_file_format='user-item_seq' \
    --user_history_file_has_header=1 \
    --user_history_file_col_names="['user_id', 'item_seq']" \
