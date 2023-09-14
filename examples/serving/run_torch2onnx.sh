#!/bin/bash 

HOME_DIR=$(eval echo ~)

LOCAL_ROOT="$HOME_DIR/UniRec"

MY_DIR=$LOCAL_ROOT

ckpt_file="path/to/SASRec.pth"
output_path="directory/to/onnx model"

cd $MY_DIR
export PYTHONPATH=$PWD

python unirec/utils/torch2onnx.py \
    --ckpt_file=$ckpt_file \
    --output_path=$output_path \
    --useful_names 'item_id' 'item_seq'