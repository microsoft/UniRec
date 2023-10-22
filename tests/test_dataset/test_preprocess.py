# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

_path = os.path.dirname(__file__)
UNIREC_PATH = os.path.abspath(os.path.join(_path, "../.."))
sys.path.append(UNIREC_PATH)

import copy
import pytest
import shutil
import datetime
from collections import *
from examples.preprocess.download_split_ml100k import prepare_ml100k
from examples.preprocess.prepare_data import process_transaction_dataset
from examples.preprocess.specific_datasets.ranker import main



CONFIG = {
    "raw_datapath": os.path.expanduser("~/.unirec/dataset/ml-100k"), 
    "outpathroot": os.path.join(UNIREC_PATH, 'tests/.temp/data'),
    "dataset_name": "ml-100k", 
    "example_yaml_file": os.path.join(UNIREC_PATH, "unirec/config/dataset/example.yaml"), 
    "index_by_zero": 0, 
    "sep": "\t" , 
    "train_file": 'train.csv', 
    "train_file_format": 'user-item', 
    "train_file_has_header": 1, 
    "train_file_col_names": "['user_id', 'item_id']", 
    "train_neg_k": 0, 
    "valid_file": 'valid.csv', 
    "valid_file_format": 'user-item', 
    "valid_file_has_header": 1, 
    "valid_file_col_names": "['user_id', 'item_id']", 
    "valid_neg_k": 0, 
    "test_file": 'test.csv', 
    "test_file_format": 'user-item', 
    "test_file_has_header": 1, 
    "test_file_col_names": "['user_id', 'item_id']", 
    "test_neg_k": 0, 
    "user_history_file": 'user_history.csv', 
    "user_history_file_format": 'user-item_seq', 
    "user_history_file_has_header": 1, 
    "user_history_file_col_names": "['user_id', 'item_seq']" , 
    "gen_text_emb": 0,
    "text_emb_size": 1024,
}


PREPARE_RAWDATA_CONFIG = {
    'data_format': 'libfm',
    'prefile': os.path.expanduser("~/.unirec/dataset/ml-100k/user_history.csv"),
    'infile_dir': os.path.expanduser("~/.unirec/dataset"),
    'outdir': os.path.join(UNIREC_PATH, 'tests/.temp/raw_datasets'),
    'n_neg_k': 20,
    'sep': '\t',
    'prefile_file_format': CONFIG['user_history_file_format']
}


def test_download_ml100k():
    flag = prepare_ml100k()
    assert flag, "ml100k dataset is not prepared successfully, this would lead to following tests' failures."


def test_preprocess_data():
    # processed data file could be used for further test pipeline
    CONFIG['gen_text_emb'] = 1
    process_transaction_dataset(CONFIG)
    assert os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'])), "processed data folder not created sucessfully"
    assert os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'data.info')), "data.info file not created sucessfully"
    assert os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'train.ftr')) or \
        os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'train.pkl')) or \
        os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'train.tsv')), "train file not created sucessfully"
    assert os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'valid.ftr')) or \
        os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'valid.pkl')) or \
        os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'valid.tsv')), "valid file not created sucessfully"
    assert os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'test.ftr')) or \
        os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'test.pkl')) or \
        os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'test.tsv')), "test file not created sucessfully"
    assert os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'user_history.ftr')) or \
        os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'user_history.pkl')) or \
        os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'user_history.tsv')), "user_history file not created sucessfully"
    assert os.path.exists(os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'text_emb.csv')), "text_emb.csv file not created sucessfully"
    if os.path.exists(os.path.join(CONFIG['raw_datapath'], 'item_meta_morec.csv')):
        shutil.copyfile(os.path.join(CONFIG['raw_datapath'], 'item_meta_morec.csv'),
                        os.path.join(CONFIG['outpathroot'], CONFIG['dataset_name'], 'item_meta_morec.csv'))


def test_preprocess_libfm_data():
    ds_name = 'ml-100k-libfm'
    pre_config = copy.deepcopy(PREPARE_RAWDATA_CONFIG)
    pre_config['infile'] = os.path.join(pre_config['infile_dir'], ds_name, f'{ds_name}.txt')
    pre_config['outdir'] = os.path.join(pre_config['outdir'], ds_name)

    main(pre_config)
    assert os.path.exists(pre_config['outdir']), "processed raw_data folder not created sucessfully"
    assert os.path.exists(pre_config['infile']), "raw_data file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'raw_data.info')), "raw_data.info file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'train.txt')), "raw train file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'valid.txt')), "raw valid file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'test.txt')), "raw test file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'user_history.txt')), "raw user_history file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'user2uid.txt')), "user to user_id file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'item2tid.txt')), "item to target_id file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'item2hid.txt')), "item to history_id file not created sucessfully"

    # processed data file could be used for further test pipeline
    config = copy.deepcopy(CONFIG)
    config['dataset_name'] = ds_name
    config['raw_datapath'] = pre_config['outdir']
    config['sep'] = " "
    config['train_file'] = 'train.txt'
    config["train_file_format"] = 'label-index_group-value_group'
    config["train_file_has_header"] = 0
    config["train_file_col_names"] = "['label', 'index_list', 'value_list']"
    config["train_neg_k"] = 0
    config["valid_file"] = 'valid.txt'
    config["valid_file_format"] = 'label-index_group-value_group'
    config["valid_file_has_header"] = 0
    config["valid_file_col_names"] = "['label', 'index_list', 'value_list']"
    config["valid_neg_k"] = 0
    config["test_file"] = 'test.txt'
    config["test_file_format"] = 'label-index_group-value_group'
    config["test_file_has_header"] = 0
    config["test_file_col_names"] = "['label', 'index_list', 'value_list']"
    config["test_neg_k"] = 0
    config["user_history_file"] = 'user_history.txt'
    config["user_history_file_format"] = 'user-item_seq'
    config["user_history_file_has_header"] = 0
    config["user_history_file_col_names"] = "['user_id', 'item_seq']" 

    process_transaction_dataset(config)
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name)), "processed data folder not created sucessfully"
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name, 'data.info')), "data.info file not created sucessfully"
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name, 'train.ftr')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'train.pkl')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'train.tsv')), "train file not created sucessfully"
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name, 'valid.ftr')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'valid.pkl')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'valid.tsv')), "valid file not created sucessfully"
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name, 'test.ftr')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'test.pkl')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'test.tsv')), "test file not created sucessfully"
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name, 'user_history.ftr')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'user_history.pkl')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'user_history.tsv')), "user_history file not created sucessfully"


def test_preprocess_rank_data():
    ds_name = 'ml-100k-rank'
    pre_config = copy.deepcopy(PREPARE_RAWDATA_CONFIG)
    pre_config['data_format'] = 'rank'
    pre_config['infile'] = os.path.join(pre_config['infile_dir'], ds_name, f'{ds_name}.txt')
    pre_config['outdir'] = os.path.join(pre_config['outdir'], ds_name)

    main(pre_config)
    assert os.path.exists(pre_config['outdir']), "processed raw_data folder not created sucessfully"
    assert os.path.exists(pre_config['infile']), "raw_data file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'train.txt')), "raw train file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'valid.txt')), "raw valid file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'test.txt')), "raw test file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'user_history.txt')), "raw user_history file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'user2uid.txt')), "user to user_id file not created sucessfully"
    assert os.path.exists(os.path.join(pre_config['outdir'], 'item2tid.txt')), "item to target_id file not created sucessfully"

    # processed data file could be used for further test pipeline
    config = copy.deepcopy(CONFIG)
    config['dataset_name'] = ds_name
    config['raw_datapath'] = os.path.join(UNIREC_PATH, 'tests/.temp/raw_datasets', ds_name)
    config['sep'] = " "
    config['train_file'] = 'train.txt'
    config["train_file_format"] = 'user-item_group-label_group'
    config["train_file_has_header"] = 0
    config["train_file_col_names"] = "['user_id', 'item_id_list', 'label_list']"
    config["train_neg_k"] = 0
    config["valid_file"] = 'valid.txt'
    config["valid_file_format"] = 'user-item_group-label_group'
    config["valid_file_has_header"] = 0
    config["valid_file_col_names"] = "['user_id', 'item_id_list', 'label_list']"
    config["valid_neg_k"] = 0
    config["test_file"] = 'test.txt'
    config["test_file_format"] = 'user-item_group-label_group'
    config["test_file_has_header"] = 0
    config["test_file_col_names"] = "['user_id', 'item_id_list', 'label_list']"
    config["test_neg_k"] = 0
    config["user_history_file"] = 'user_history.txt'
    config["user_history_file_format"] = 'user-item_seq'
    config["user_history_file_has_header"] = 0
    config["user_history_file_col_names"] = "['user_id', 'item_seq']" 

    process_transaction_dataset(config)
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name)), "processed data folder not created sucessfully"
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name, 'data.info')), "data.info file not created sucessfully"
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name, 'train.ftr')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'train.pkl')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'train.tsv')), "train file not created sucessfully"
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name, 'valid.ftr')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'valid.pkl')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'valid.tsv')), "valid file not created sucessfully"
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name, 'test.ftr')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'test.pkl')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'test.tsv')), "test file not created sucessfully"
    assert os.path.exists(os.path.join(config['outpathroot'], ds_name, 'user_history.ftr')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'user_history.pkl')) or \
        os.path.exists(os.path.join(config['outpathroot'], ds_name, 'user_history.tsv')), "user_history file not created sucessfully"


if __name__ == "__main__":
    test_download_ml100k()
    test_preprocess_data()
    # test_preprocess_libfm_data()
    # test_preprocess_rank_data()