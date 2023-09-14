#!/bin/bash 

## No arguments, which means local execution
LOCAL_ROOT='/home/v-huangxu/work/UniRec' 

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output" 
MODEL_NAME='MF' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec, LKNN, FASTLKNN]
loss_type='bpr' # [bce, bpr, softmax]
DATASET_NAME="amazon-electronics" # "xas_1111" "xas_1114"
max_seq_len=20
verbose=2

cd $MY_DIR
export PYTHONPATH=$PWD 

# overall config
DATA_TYPE='BaseDataset'  # BaseDataset SeqRecDataset

# train
learning_rate=0.001
test_protocol='one_vs_all'  # 'one_vs_k' 'one_vs_all' 'session_aware'

ngroup=10
item_price_filename="item_price.tsv"
item_category_filename="item_category.tsv"

currentTime=`date "+%Y-%m-%d_%H%M%S"`
exp_name="MoRec-BaseModel"
checkpoint_dir=$DATASET_NAME"_"$MODEL_NAME"_"$loss_type"_"$currentTime

# for  MODEL_NAME in 'AvgHist' 'AttHist' 'MF' 'SVDPlusPlus' 'GRU4Rec' 'SASRec'
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
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --learning_rate=$learning_rate \
    --dropout_prob=0.0 \
    --embedding_size=64 \
    --hidden_size=64 \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=200  \
    --early_stop=-1 \
    --batch_size=512 \
    --n_sample_neg_train=10 \
    --neg_by_pop_alpha=1.0 \
    --valid_protocol=$test_protocol \
    --test_protocol=$test_protocol \
    --grad_clip_value=-1 \
    --weight_decay=1e-6 \
    --history_mask_mode='autoregressive' \
    --user_history_filename="user_history" \
    --metrics="['hit@10;20', 'ndcg@10;20', 'rhit@10;20', 'rndcg@10;20']" \
    --key_metric="ndcg@10" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --item_price_filename=$item_price_filename \
    --item_category_filename=$item_category_filename \
    --checkpoint_dir=$checkpoint_dir \
    --exp_name=$exp_name


echo ">>> Step1: Pretrain step completed."


# required arguments
enable_morec=1
ngroup=10
alpha=0.01
expect_loss=0.25
beta_min=0.1
beta_max=1.5
K_p=0.05
K_i=0.001
objective_weights="[0.1,0.1,0.8]"   # weight for objectives expect accuracy

# well-trained model
model_file="$ALL_RESULTS_ROOT/train/$checkpoint_dir/$MODEL_NAME-$MODEL_NAME-$exp_name.pth"

exp_name="MoRec-FinetuneModel"
checkpoint_dir=$DATASET_NAME"_"$MODEL_NAME"_"$loss_type"_MoRecFinetune_"$currentTime

### train ###################################

# python -m debugpy --listen 9999 --wait-for-client main/main.py \
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --epochs=50  \
    --early_stop=-1 \
    --metrics="['hit@10', 'ndcg@10', 'rhit@10', 'rndcg@10', 'pop-kl@10', 'least-misery']" \
    --key_metric="ndcg@10" \
    --enable_morec=$enable_morec \
    --morec_objectives 'fairness' 'alignment' 'revenue' \
    --morec_ngroup=$ngroup \
    --morec_alpha=$alpha \
    --morec_expect_loss=$expect_loss \
    --morec_beta_min=$beta_min \
    --morec_beta_max=$beta_max \
    --morec_K_p=$K_p \
    --morec_K_i=$K_i \
    --morec_objective_weights=$objective_weights \
    --item_price_filename=$item_price_filename \
    --item_category_filename=$item_category_filename \
    --model_file=$model_file \
    --checkpoint_dir=$checkpoint_dir \
    --exp_name=$exp_name

echo ">>> Step2: MoRec finetune completed."