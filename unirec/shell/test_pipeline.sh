#!/bin/bash 
 
## When you add new models or new features to the library, 
## don't forget to add a running job in this script and lauch the whole shell to verify that the pipeline is running normally.

 
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple 
White='\033[0;37m'        # White
NC='\033[0m' # No Color


LOCAL_ROOT='/media/xreco/MSRA/jianxun/UniRec' 
DATASET_NAME="xas_1109_small" # "xas_1111"

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output_integration_test" 

cd $MY_DIR
export PYTHONPATH=$PWD 
 
DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset 
learning_rate=0.002 
test_protocol='one_vs_all'  # 'one_vs_k' 'one_vs_all' 'session_aware'
history_mask_mode='unorder' # 'autoregressive'
max_seq_len=10
verbose=2 
neg_by_pop_alpha=0

idx=0
for MODEL_NAME in 'AvgHist' 'MF' 'SASRec'
do
for loss_type in 'bce' 'bpr' 'softmax' 
do
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT

let "idx+=1" 
echo ''
echo -e "${Green}Run config $idx:${NC}"

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
    --epochs=3  \
    --early_stop=1 \
    --batch_size=512 \
    --n_sample_neg_train=2 \
    --n_sample_neg_valid=9 \
    --test_protocol=$test_protocol \
    --grad_clip_value=0.1 \
    --weight_decay=1e-6 \
    --history_mask_mode=$history_mask_mode \
    --user_history_filename="user_history" \
    --metrics="['group_auc', 'hit@1;2;3', 'ndcg@1;2;3']" \
    --key_metric="ndcg@3" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --neg_by_pop_alpha=$neg_by_pop_alpha
done
done


### testing neg_by_pop_alpha
for neg_by_pop_alpha in -0.5 0.5 1.5
do
let "idx+=1" 
echo ''
echo -e "${Green}Run config $idx:${NC}"
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/AvgHist"
mkdir -p $ALL_RESULTS_ROOT
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model='AvgHist' \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --learning_rate=0.01 \
    --dropout_prob=0.0 \
    --embedding_size=32 \
    --hidden_size=32 \
    --use_pre_item_emb=0 \
    --loss_type='softmax' \
    --max_seq_len=10 \
    --has_user_bias=1 \
    --has_item_bias=1 \
    --epochs=1  \
    --early_stop=1 \
    --batch_size=512 \
    --n_sample_neg_train=2 \
    --n_sample_neg_valid=9 \
    --test_protocol='one_vs_all' \
    --grad_clip_value=0.1 \
    --weight_decay=1e-6 \
    --history_mask_mode='autoregressive' \
    --user_history_filename="user_history" \
    --metrics="['group_auc', 'hit@1;2;3', 'ndcg@1;2;3']" \
    --key_metric="ndcg@3" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=2 \
    --neg_by_pop_alpha=$neg_by_pop_alpha
done

### testing session_aware and autoregressive 
let "idx+=1" 
echo ''
echo -e "${Green}Run config $idx:${NC}"
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/AvgHist"
mkdir -p $ALL_RESULTS_ROOT
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model='AvgHist' \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --learning_rate=0.01 \
    --dropout_prob=0.0 \
    --embedding_size=32 \
    --hidden_size=32 \
    --use_pre_item_emb=0 \
    --loss_type='softmax' \
    --max_seq_len=10 \
    --has_user_bias=1 \
    --has_item_bias=1 \
    --epochs=1  \
    --early_stop=1 \
    --batch_size=512 \
    --n_sample_neg_train=2 \
    --n_sample_neg_valid=9 \
    --test_protocol='session_aware' \
    --grad_clip_value=0.1 \
    --weight_decay=1e-6 \
    --history_mask_mode='autoregressive' \
    --user_history_filename="user_history" \
    --metrics="['group_auc', 'hit@1;2;3', 'ndcg@1;2;3']" \
    --key_metric="ndcg@3" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=2 \
    --neg_by_pop_alpha=0

############ MultiVAE
let "idx+=1" 
echo ''
echo -e "${Green}Run config $idx:${NC}"

MODEL_NAME='MultiVAE'
DATA_TYPE='AERecDataset'  # BaseDataset SeqRecDataset
test_protocol='one_vs_all'  #'one_vs_all' 'session_aware'
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT 
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
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=3  \
    --early_stop=1 \
    --batch_size=512 \
    --n_sample_neg_train=2 \
    --n_sample_neg_valid=9 \
    --test_protocol=$test_protocol \
    --grad_clip_value=0.1 \
    --weight_decay=1e-6 \
    --history_mask_mode=$history_mask_mode \
    --user_history_filename="user_history" \
    --metrics="['group_auc', 'hit@1;2;3', 'ndcg@1;2;3']" \
    --key_metric="ndcg@3" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --neg_by_pop_alpha=$neg_by_pop_alpha

############ SAR
# overall config
for MODEL_NAME in 'SAR' 'EASE' #'AdmmSLIM' 'SLIM'
do
let "idx+=1" 
echo ''
echo -e "${Green}Run config $idx:${NC}" 
DATA_TYPE='AERecDataset'  # BaseDataset SeqRecDataset
test_protocol='one_vs_all'  #'one_vs_all' 'session_aware'
 
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT 
### do train and evaluation on test test
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --test_protocol=$test_protocol \
    --history_mask_mode='unorder' \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq-time_seq" \
    --metrics="['group_auc', 'hit@5;10;20', 'ndcg@5;10;20']" \
    --key_metric="ndcg@5" \
    --verbose=2 \
    --epochs=1  \
    --num_workers_test=0
done


### testing CCL loss
let "idx+=1" 
echo ''
echo -e "${Green}Run config $idx:${NC}"
MODEL_NAME='AvgHist'
DATA_TYPE='SeqRecDataset'
loss_type='ccl'
ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --learning_rate=0.01 \
    --dropout_prob=0.0 \
    --embedding_size=32 \
    --hidden_size=32 \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=10 \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=1  \
    --early_stop=1 \
    --batch_size=512 \
    --n_sample_neg_train=100 \
    --n_sample_neg_valid=9 \
    --test_protocol='one_vs_all' \
    --grad_clip_value=0.1 \
    --weight_decay=1e-6 \
    --history_mask_mode='autoregressive' \
    --user_history_filename="user_history" \
    --metrics="['group_auc', 'hit@1;2;3', 'ndcg@1;2;3']" \
    --key_metric="ndcg@3" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=2 \
    --distance_type='cosine' \
    --ccl_w=20 \
    --ccl_m=0.4 \