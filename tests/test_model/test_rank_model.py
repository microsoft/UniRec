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


TOL = 0.2
ABS_TOL = 0.2

GLOBAL_CONF = {
    'config_dir': f"{os.path.join(UNIREC_PATH, 'unirec', 'config')}",
    'exp_name': 'pytest',
    'checkpoint_dir': f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
    'model': '',
    'base_model': 'GRU',
    'train_type': 'Ada-Ranker',
    'dataloader': '',
    'dataset': '',
    'dataset_path': os.path.join(UNIREC_PATH, 'tests/.temp/data'),
    'output_path': '',
    'learning_rate': 0.0008787070324991168,
    'use_pre_item_emb': 0,
    'loss_type': 'bce',
    'optimizer': 'adam',
    'has_user_bias': 0,
    'has_item_bias': 0,
    'epochs': 20,
    'batch_size': 1024,
    'n_sample_neg_train': 0,
    'valid_protocol': 'one_vs_k',
    'test_protocol': 'one_vs_k',
    'grad_clip_value': 10,
    'user_history_filename': "user_history",
    'user_history_file_format': "user-item_seq",
    'history_mask_mode': 'autoagressive',
    'group_size': -1,
    'metrics': "['auc','group_auc']",
    'key_metric': "auc",
    'shuffle_train': True,
    'early_stop': 5,
    'embedding_size': 32,
    'num_workers': 4,
    'num_workers_test': 0,
    'verbose': 2,
    'neg_by_pop_alpha': 0.0,
    'scheduler_factor': 0.5,
    'n_layers': 3,
    'n_heads': 8,
    'inner_size': 64,
    'hidden_dropout_prob': 0.11175639972166328,
    'attn_dropout_prob': 0.22652963648975333,
    'max_seq_len': 10,
    'seed': 2023,
    'freeze': 0,
}

RANK_SGD_MODELS = ["FM", "BST", "AdaRanker"]  # models optimized by SGD
EXPECTED_METRICS = {
    "FM": {"group_auc": 0.79453, "auc": 0.81613},
    "BST": {"group_auc": 0.83397, "auc": 0.85084},
    "AdaRanker": {"group_auc": 0.78692, "auc": 0.82794}
}
MODEL2DATASET = {
    "FM": "ml-100k-libfm",
    "BST": "ml-100k-rank",
    "AdaRanker": "ml-100k-adaranker"
}
MODEL2DATALOADER = {
    "FM": "RankDataset",
    "BST": "SeqRecDataset",
    "AdaRanker": "SeqRecDataset"
}


# >>>>>> Test train pipeline of FM model and check the performance
# Note: this train test instance should be put in the first place because the model checkpoint files generated here are required in following tests 
@pytest.mark.parametrize(
        "models, expected_values",
        [
            (
                RANK_SGD_MODELS,
                EXPECTED_METRICS
            )
        ]
)
def test_train_pipeline(models, expected_values):
    for model in models:
        data = MODEL2DATASET[model]
        config = copy.deepcopy(GLOBAL_CONF)
        config['task'] = 'train'
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['dataset'] = data
        config['model'] = model
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        config['dataloader'] = MODEL2DATALOADER[model]
        if model == 'FM':
            config['epochs'] = 10
            config['learning_rate'] = 0.001
            config['embedding_size'] = 80
            config['group_size'] = 21
        elif model == 'AdaRanker':
            config['epochs'] = 100
            config['learning_rate'] = 0.001
            config['embedding_size'] = 64
            config['key_metric'] = 'group_auc'
            config['dropout_prob'] = 0.6
            config['max_seq_len'] = 10
            config['use_pre_item_emb'] = 1
            config['batch_size'] = 256
            config['early_stop'] = 15
            config['item_emb_path'] = os.path.join(UNIREC_PATH, 'tests/.temp/raw_datasets/ml-100k-adaranker/item_emb_64.txt')
        exp_value = expected_values[model]
        result = main.run(config)
        for k, v in exp_value.items():
            assert result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL), "performance of {} not correct".format(model)


# >>>>>> Test evaluate task of FM model and check the performance
@pytest.mark.parametrize(
        "models, expected_values",
        [
            (
                RANK_SGD_MODELS,
                EXPECTED_METRICS
            )
        ]
)
def test_eval_pipeline(models, expected_values):
    for model in models:
        data = MODEL2DATASET[model]
        config = copy.deepcopy(GLOBAL_CONF)
        config['task'] = 'test'
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['dataset'] = data
        config['model'] = model
        config['dataloader'] = MODEL2DATALOADER[model]
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        checkpoint_dir = os.path.join(config['output_path'], config['checkpoint_dir'])
        config['model_file'] = os.path.join(checkpoint_dir, f"{model}-{config['exp_name']}.pth")
        if model == 'FM':
            config['epochs'] = 10
            config['learning_rate'] = 0.001
            config['embedding_size'] = 80
            config['group_size'] = 21
        elif model == 'AdaRanker':
            config['epochs'] = 100
            config['learning_rate'] = 0.001
            config['embedding_size'] = 64
            config['key_metric'] = 'group_auc'
            config['dropout_prob'] = 0.6
            config['max_seq_len'] = 10
            config['use_pre_item_emb'] = 1
            config['batch_size'] = 256
            config['early_stop'] = 15
            config['item_emb_path'] = os.path.join(UNIREC_PATH, 'tests/.temp/raw_datasets/ml-100k-adaranker/item_emb_64.txt')
        exp_value = expected_values[model]
        result = main.run(config)
        for k, v in exp_value.items():
            assert result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL), "performance of {} not correct".format(model)


# >>>>>> Test infer task of FM model and check the shapes of predicted scores
@pytest.mark.parametrize(
        "models",
        [
            RANK_SGD_MODELS
        ]
)
def test_infer_pipeline(models):
    shape = set()
    for model in models:
        data = MODEL2DATASET[model]
        config = copy.deepcopy(GLOBAL_CONF)
        config['task'] = 'infer'
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['dataset'] = data
        config['model'] = model
        config['dataloader'] = MODEL2DATALOADER[model]
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        checkpoint_dir = os.path.join(config['output_path'], config['checkpoint_dir'])
        config['model_file'] = os.path.join(checkpoint_dir, f"{model}-{config['exp_name']}.pth")
        if model == 'FM':
            config['epochs'] = 10
            config['learning_rate'] = 0.001
            config['embedding_size'] = 80
            config['group_size'] = 21
        elif model == 'AdaRanker':
            config['epochs'] = 100
            config['learning_rate'] = 0.001
            config['embedding_size'] = 64
            config['key_metric'] = 'group_auc'
            config['dropout_prob'] = 0.6
            config['max_seq_len'] = 10
            config['use_pre_item_emb'] = 1
            config['batch_size'] = 256
            config['early_stop'] = 15
            config['item_emb_path'] = os.path.join(UNIREC_PATH, 'tests/.temp/raw_datasets/ml-100k-adaranker/item_emb_64.txt')
        result = main.run(config)
        shape.add(result.shape)

    assert len(shape) == 2, "shapes of scores infered by those models are different"


if __name__ == "__main__":
    pytest.main(["test_rank_model.py", "-s"])