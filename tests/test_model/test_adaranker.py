# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

_path = os.path.dirname(__file__)
UNIREC_PATH = os.path.abspath(os.path.join(_path, "../.."))
sys.path.append(UNIREC_PATH)

import copy
import pytest
import datetime
from collections import *
from unirec.main import main


TOL = 0.1
ABS_TOL = 0.1

GLOBAL_CONF = {
    'config_dir': f"{os.path.join(UNIREC_PATH, 'unirec', 'config')}",
    'exp_name': 'pytest',
    'checkpoint_dir': f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
    'model': 'AdaRanker',
    'base_model': 'GRU',
    'train_type': '',
    'dataloader': 'SeqRecDataset',
    'dataset': 'ml-100k-rank',
    'dataset_path': os.path.join(UNIREC_PATH, 'tests/.temp/data/ml-100k-rank'),
    'output_path': '',
    'learning_rate': 0.001,
    'use_pre_item_emb': 1,
    'item_emb_path': os.path.join(UNIREC_PATH, 'tests/.temp/raw_datasets/ml-100k-rank/item_emb_32.txt'),
    'loss_type': 'bce',
    'optimizer': 'adam',
    'has_user_bias': 0,
    'has_item_bias': 0,
    'epochs': 100,
    'batch_size': 1024,
    'n_sample_neg_train': 0,
    'valid_protocol': 'one_vs_k',
    'test_protocol': 'one_vs_k',
    'user_history_filename': "user_history",
    'user_history_file_format': "user-item_seq",
    'history_mask_mode': 'autoagressive',
    'group_size': -1,
    'metrics': "['group_auc','auc']",
    'key_metric': "group_auc",
    'shuffle_train': True,
    'early_stop': 15,
    'embedding_size': 32,
    'num_workers': 4,
    'num_workers_test': 0,
    'verbose': 2,
    'neg_by_pop_alpha': 0.0,
    'scheduler_factor': 0.5,
    'n_layers': 3,
    'n_heads': 8,
    'inner_size': 64,
    'hidden_dropout_prob': 0.5,
    'attn_dropout_prob': 0.5,
    'dropout_prob': 0.4,
    'max_seq_len': 20,
    'seed': 2023,
    'task': 'train',
    'freeze': 0
}

TRAIN_TYPES = ["Base", "Ada-Ranker"]  # train type of AdaRanker
EXPECTED_METRICS = {
    "Base": {"group_auc": 0.78812, "auc": 0.80083},
    "Ada-Ranker": {"group_auc": 0.78296, "auc": 0.79590},
    "Finetune": {"group_auc": 0.78610, "auc": 0.80376}
}


# >>>>>> Test train pipeline of AdaRanker model and check the performance
# Note: this train test instance should be put in the first place because the model checkpoint files generated here are required in following tests 
@pytest.mark.parametrize(
        "data, model, train_types, expected_values",
        [
            (
                "ml-100k-rank",
                "AdaRanker",
                TRAIN_TYPES,
                EXPECTED_METRICS
            )
        ]
)
def test_train_end_to_end_pipeline(data, model, train_types, expected_values):
    for train_type in train_types:
        config = copy.deepcopy(GLOBAL_CONF)
        config['train_type'] = train_type
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}_{train_type}')
        exp_value = expected_values[train_type]
        result = main.run(config)
        for k, v in exp_value.items():
            assert result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL), "performance of {} not correct".format(train_type)


# >>>>>> Test fintune pipeline of AdaRanker model and check the performance
@pytest.mark.parametrize(
        "data, model, expected_values",
        [
            (
                "ml-100k-rank",
                "AdaRanker",
                EXPECTED_METRICS
            )
        ]
)
def test_fintune_pipeline(data, model, expected_values):
    config = copy.deepcopy(GLOBAL_CONF)
    config['train_type'] = "Ada-Ranker"
    config['freeze'] = 1
    config['use_pre_item_emb'] = 0
    config['load_pretrained_model'] = 1
    # config['learning_rate'] = 0.001
    directory = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}_Base')
    pretrained_model_dir = next((name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))), None)
    config['model_file'] = '' if pretrained_model_dir is None else os.path.join(directory, pretrained_model_dir, f'{model}-pytest.pth')
    config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}_Fintune')
    exp_value = expected_values["Finetune"]
    result = main.run(config)
    for k, v in exp_value.items():
        assert result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL), "performance of Finetune not correct"


if __name__ == "__main__":
    pytest.main(["test_adaranker.py", "-s"])