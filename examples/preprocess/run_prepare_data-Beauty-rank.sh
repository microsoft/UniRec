RAW_DATA_FILE="/path/to/Beauty/Beauty.txt"
RAW_DATA_DIR="/path/to/raw_datasets"

ROOT_DIR='/path/to/UniRec'
MY_DIR=$ROOT_DIR
DATA_ROOT="$ROOT_DIR/data"
OUTPUT_ROOT="$ROOT_DIR/output"


dataset_name='Beauty-rank' ## gowalla amazon-book yelp2018

export PYTHONPATH=$MY_DIR

raw_datapath="$RAW_DATA_DIR/$dataset_name" 
dataset_outpathroot=$DATA_ROOT
example_yaml_file="$MY_DIR/unirec/config/dataset/example.yaml"

# group_size=21
# n_neg_k=$((group_size-1)) # if use group_size

group_size=-1
n_neg_k=20


cd $MY_DIR"/preprocess"
# run ranker.py to get rank data in T4 data format from user history
python specific_datasets/ranker.py \
    --data_format='rank' \
    --infile=$RAW_DATA_FILE \
    --outdir=$raw_datapath \
    --n_neg_k=$n_neg_k


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
