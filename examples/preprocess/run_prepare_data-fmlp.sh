RAW_DATA_FILE="/path/to/datasets/data/Beauty.txt"
RAW_DATA_DIR="/path/to/fmlp"

ROOT_DIR='/home/jialia/adaretriever/UniRec'
MY_DIR=$ROOT_DIR
DATA_ROOT="$ROOT_DIR/data"
OUTPUT_ROOT="$ROOT_DIR/output"


dataset_name='Beauty' ## gowalla amazon-book yelp2018

export PYTHONPATH=$MY_DIR

raw_datapath="$RAW_DATA_DIR/$dataset_name" 
dataset_outpathroot=$DATA_ROOT
example_yaml_file="$MY_DIR/unirec/config/dataset/example.yaml"


cd $MY_DIR"/preprocess"
python specific_datasets/fmlp.py \
    --infile=$RAW_DATA_FILE \
    --outdir=$raw_datapath


python prepare_data.py \
    --raw_datapath=$raw_datapath \
    --outpathroot=$dataset_outpathroot \
    --dataset_name=$dataset_name \
    --example_yaml_file=$example_yaml_file \
    --index_by_zero=0 \
    --sep=" "  \
    --train_file='train.txt'\
    --train_file_format='user-item_seq' \
    --train_file_col_names="['user_id', 'item_seq']" \
    --train_file_has_header=0 \
    --train_neg_k=0 \
    --valid_file='valid.txt'\
    --valid_file_format='user-item' \
    --valid_file_col_names="['user_id', 'item_id']" \
    --valid_file_has_header=0 \
    --valid_neg_k=0 \
    --test_file='test.txt'\
    --test_file_format='user-item' \
    --test_file_col_names="['user_id', 'item_id']" \
    --test_file_has_header=0 \
    --test_neg_k=0 \
    --user_history_file='user_history.txt'\
    --user_history_file_format='user-item_seq' \
    --user_history_file_has_header=0 \
    --user_history_file_col_names="['user_id', 'item_seq']" 

    