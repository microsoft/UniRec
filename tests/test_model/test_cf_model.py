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
from unirec.main import main, infer_embedding


TOL = 0.05
ABS_TOL = 0.05

GLOBAL_CONF = {
    'config_dir': f"{os.path.join(UNIREC_PATH, 'unirec', 'config')}",
    'exp_name': 'pytest',
    'checkpoint_dir': f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
    'model': '',
    'dataloader': '',
    'dataset': '',
    'dataset_path': os.path.join(UNIREC_PATH, 'tests/.temp/data'),
    'output_path': '',
    'learning_rate': 0.001,
    'dropout_prob': 0.0,
    'embedding_size': 32,
    'hidden_size': 32,
    'use_pre_item_emb': 0,
    'loss_type': 'bce',
    'max_seq_len': 10,
    'has_user_bias': 1,
    'has_item_bias': 1,
    'epochs': 3,
    'early_stop': -1,
    'batch_size': 512,
    'n_sample_neg_train': 9,
    'valid_protocol': 'one_vs_all',
    'test_protocol': 'one_vs_all',
    'grad_clip_value': 0.1,
    'weight_decay': 1e-6,
    'history_mask_mode': 'autoagressive',
    'user_history_filename': "user_history",
    'metrics': "['hit@5;10', 'ndcg@5;10']",
    'key_metric': "ndcg@5",
    'num_workers': 4,
    'num_workers_test': 0,
    'verbose': 2,
    'neg_by_pop_alpha': 0.0,
    'conv_size': 10, # for ConvFormer-series
}

CF_SGD_MODELS = ["MultiVAE", "MF"]  # models optimized by SGD
CF_SOLVE_MODELS = ["SLIM", "AdmmSLIM", "SAR", "EASE", "UserCF"] # models optimized by other optimization algorithms, such as SAR, EASE
EXPECTED_METRICS = {
                        'MultiVAE': {'hit@5': 0.01065, 'ndcg@5': 0.00890},
                        'MF': {'hit@5': 0.04686, 'ndcg@5': 0.02997},
                        'SLIM': {'hit@5': 0.07135, 'ndcg@5': 0.04876},
                        'AdmmSLIM': {'hit@5': 0.08946, 'ndcg@5': 0.05517},
                        'SAR': {'hit@5': 0.06816, 'ndcg@5': 0.04373},
                        'EASE': {'hit@5': 0.08200, 'ndcg@5': 0.05530},
                        'UserCF': {'hit@5': 0.06283, 'ndcg@5': 0.04186},
                    }
MODEL2DATALOADER = {
    "MF": "BaseDataset",
    "MultiVAE": "AERecDataset", 
    "SAR": "AERecDataset",
    "EASE": "AERecDataset",
    "UserCF": "AERecDataset",
    "SLIM": "AERecDataset",
    "AdmmSLIM": "AERecDataset"
}



# >>>>>> Test train pipeline of CF models and check the performance
# Note: this train test instance should be put in the first place because the model checkpoint files generated here are required in following tests 
@pytest.mark.parametrize(
        "data, models, expected_values",
        [
            (
                "ml-100k",
                CF_SGD_MODELS + CF_SOLVE_MODELS,
                EXPECTED_METRICS
            )
        ]
)
def test_train_pipeline(data, models, expected_values):
    all_result = {}
    # finish all training first for following evaluation and infer test
    for model in models:
        config = copy.deepcopy(GLOBAL_CONF)
        config['task'] = 'train'
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['dataset'] = data
        config['model'] = model
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        config['dataloader'] = MODEL2DATALOADER[model]
        exp_value = expected_values[model]
        result = main.run(config)
        all_result[model] = result

    # check the performance
    failed_models = []
    for model in models:
        result = all_result[model]
        exp_value = expected_values[model]
        for k, v in exp_value.items():
            if not result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL):
                failed_models.append(model)
                break
    assert len(failed_models)==0, f"performance of [{', '.join(failed_models)}] not correct."


# >>>>>> Test evaluate task of CF models and check the performance
@pytest.mark.parametrize(
        "data, models, expected_values",
        [
            (
                "ml-100k",
                CF_SGD_MODELS + CF_SOLVE_MODELS,
                EXPECTED_METRICS
            )
        ]
)
def test_eval_pipeline(data, models, expected_values):
    for model in models:
        config = copy.deepcopy(GLOBAL_CONF)
        config['task'] = 'test'
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['dataset'] = data
        config['model'] = model
        config['dataloader'] = MODEL2DATALOADER[model]
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        checkpoint_dir = os.path.join(config['output_path'], config['checkpoint_dir'])
        config['model_file'] = os.path.join(checkpoint_dir,  f"{model}-{config['exp_name']}.pth")
        exp_value = expected_values[model]
        result = main.run(config)
        for k, v in exp_value.items():
            assert result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL), "performance of {} not correct".format(model)



# >>>>>> Test infer task of CF models and check the shapes of predicted scores
@pytest.mark.parametrize(
        "data, models",
        [
            (
                "ml-100k",
                CF_SGD_MODELS + CF_SOLVE_MODELS,
            )
        ]
)
def test_infer_pipeline(data, models):
    shape = set()
    for model in models:
        config = copy.deepcopy(GLOBAL_CONF)
        config['task'] = 'infer'
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['dataset'] = data
        config['model'] = model
        config['dataloader'] = MODEL2DATALOADER[model]
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        checkpoint_dir = os.path.join(config['output_path'], config['checkpoint_dir'])
        config['model_file'] = os.path.join(checkpoint_dir,  f"{model}-{config['exp_name']}.pth")
        result = main.run(config)
        shape.add(result.shape)

    assert len(shape) == 1, "shapes of scores infered by those models are different"
    


# >>>>>> Test embedding task of CF models optimized by SGD and check the shapes of returned embeddings
@pytest.mark.parametrize(
        "data, models",
        [
            (
                "ml-100k",
                CF_SGD_MODELS
            )
        ]
)
def test_item_embedding_pipeline(data, models):
    for model in models:
        config = copy.deepcopy(GLOBAL_CONF)
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['node_type'] = 'item'
        config['output_emb_file'] = os.path.join(UNIREC_PATH, f"tests/.temp/output/{data}/{model}/item_embedding/result_item_embedding.txt")
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        checkpoint_dir = os.path.join(config['output_path'], config['checkpoint_dir'])
        config['model_file'] = os.path.join(checkpoint_dir, f"{model}-{config['exp_name']}.pth")
        item_ids, item_embs = infer_embedding.run(config)
        assert item_embs.shape[0] == len(item_ids), f"for model {model}, number of item embeddings is not equal to number of items"
        assert item_embs.shape[1] == config['embedding_size'], \
            f"for model {model}, the -1 dimension of generated item embedding is not equal to `config['embedding_size']`"

    

@pytest.mark.parametrize(
        "data, models",
        [
            (
                "ml-100k",
                CF_SGD_MODELS
            )
        ]
)
def test_user_embedding_pipeline(data, models):
    for model in models:
        config = copy.deepcopy(GLOBAL_CONF)
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['node_type'] = 'user'
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        config['output_emb_file'] = os.path.join(config['output_path'], "item_embedding/result_user_embedding.txt")
        checkpoint_dir = os.path.join(config['output_path'], config['checkpoint_dir'])
        config['model_file'] = os.path.join(checkpoint_dir,  f"{model}-{config['exp_name']}.pth")
        user_ids, user_embs = infer_embedding.run(config)
        assert user_embs.shape[0] == len(user_ids), f"for model {model}, number of user embeddings is not equal to number of users"
        assert user_embs.shape[1] == config['embedding_size'], \
            f"for model {model}, the -1 dimension of generated user embedding is not equal to `config['embedding_size']`"
        

if __name__ == "__main__":
    pytest.main(["test_cf_model.py", "-s"])