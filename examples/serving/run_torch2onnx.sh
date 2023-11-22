#!/bin/bash 

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
HOME_DIR=$(eval echo ~)

LOCAL_ROOT="$HOME_DIR/UniRec"

ckpt_file="path/to/SASRec.pth"
output_path="directory/to/onnx model"


###############################################################################################
############################## default parameters for local run ###############################
###############################################################################################
MY_DIR=$LOCAL_ROOT


cd $MY_DIR
export PYTHONPATH=$PWD

python unirec/utils/torch2onnx.py \
    --ckpt_file=$ckpt_file \
    --output_path=$output_path \
    --useful_names 'item_id' 'item_seq'