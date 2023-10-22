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
    'dataloader': 'SeqRecDataset',
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

SEQ_MODELS = ["SVDPlusPlus", "FASTConvFormer", "ConvFormer", "SASRec", "AvgHist", "GRU4Rec", "AttHist"]  # Each test is ordered according to the list
LOSS_TYPES = ["bce", "bpr", "softmax", "ccl", "fullsoftmax"]
EXPECTED_METRICS = {
                        'SVDPlusPlus': {'hit@5': 0.04792, 'ndcg@5': 0.03394},
                        'FASTConvFormer': {'hit@5': 0.05005, 'ndcg@5': 0.03355},
                        'ConvFormer': {'hit@5': 0.05005, 'ndcg@5': 0.03538},
                        'SASRec': {'hit@5': 0.04792, 'ndcg@5': 0.03184},
                        'AvgHist': {'hit@5': 0.05005, 'ndcg@5': 0.03423},
                        'GRU4Rec': {'hit@5': 0.04686, 'ndcg@5': 0.03197},
                        'AttHist': {'hit@5': 0.04686, 'ndcg@5': 0.03221},
                        'SASRec_bce': {'hit@5': 0.04792, 'ndcg@5': 0.03184},
                        'SASRec_bpr': {'hit@5': 0.04686, 'ndcg@5': 0.03122},
                        'SASRec_softmax': {'hit@5': 0.04686, 'ndcg@5': 0.03066},
                        'SASRec_ccl': {'hit@5': 0.02449, 'ndcg@5': 0.01318},
                        'SASRec_fullsoftmax': {'hit@5': 0.04792, 'ndcg@5': 0.03155},
                        'SASRec_with_text_emb': {'hit@5': 0.04686, 'ndcg@5': 0.03219},
                    }



# >>>>>> Test train pipeline of sequential models and check the performance
# Note: the test instance should be put in the first place because the model checkpoint files generated here are required in following tests 
@pytest.mark.parametrize(
        "data, models, expected_values",
        [
            (
                "ml-100k",
                SEQ_MODELS,
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
        result = main.run(config)
        all_result[model] = result

    # check the performance
    failed_models = []
    for model in models:
        exp_value = expected_values[model]
        result = all_result[model]
        for k, v in exp_value.items():
            if not result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL):
                failed_models.append(model)
                break
    assert len(failed_models)==0, f"performance of [{', '.join(failed_models)}] not correct."

@pytest.mark.parametrize(
        "data, models, expected_values",
        [
            (
                "ml-100k",
                ["SASRec"],
                EXPECTED_METRICS
            )
        ]
)
def test_text_emb_pipeline(data, models, expected_values):
    all_result = {}
    # finish all training first for following evaluation and infer test
    for model in models:
        config = copy.deepcopy(GLOBAL_CONF)
        config['task'] = 'train'
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['dataset'] = data
        config['model'] = model
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        config['use_text_emb'] = 1
        config['text_emb_path'] = os.path.join(config['dataset_path'], 'text_emb.csv')
        config['text_emb_size'] = 1024
        result = main.run(config)
        all_result[f"{model}_with_text_emb"] = result

    # check the performance
    failed_models = []
    for model in models:
        exp_value = expected_values[f"{model}_with_text_emb"]
        result = all_result[f"{model}_with_text_emb"]
        for k, v in exp_value.items():
            if not result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL):
                failed_models.append(f"{model}_with_text_emb")
                break
    assert len(failed_models)==0, f"performance of [{', '.join(failed_models)}] not correct."


@pytest.mark.parametrize(
        "data, model, loss_types, expected_values",
        [
            (
                "ml-100k",
                "SASRec",
                LOSS_TYPES,
                EXPECTED_METRICS
            )
        ]
)
def test_losstype_pipeline(data, model, loss_types, expected_values):
    all_result = {}
    # finish all training first for following evaluation and infer test
    for loss_type in loss_types:
        config = copy.deepcopy(GLOBAL_CONF)
        config['task'] = 'train'
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['dataset'] = data
        config['model'] = model
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}_{loss_type}')
        config['loss_type'] = loss_type
        config['early_stop'] = -1
        if loss_type == 'fullsoftmax':
            config['n_sample_neg_train'] = 0
        result = main.run(config)
        all_result[f"{model}_{loss_type}"] = result

    # check the performance
    failed_models = []
    for loss_type in loss_types:
        result = all_result[f"{model}_{loss_type}"]
        exp_value = expected_values[f"{model}_{loss_type}"]
        for k, v in exp_value.items():
            if not result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL):
                failed_models.append(model)
                break
    assert len(failed_models)==0, f"performance of [{', '.join(failed_models)}] not correct."


# >>>>>> Test evaluate task of sequential models and check the performance
@pytest.mark.parametrize(
        "data, models, expected_values",
        [
            (
                "ml-100k",
                SEQ_MODELS,
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
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        checkpoint_dir = os.path.join(config['output_path'], config['checkpoint_dir'])
        config['model_file'] = os.path.join(checkpoint_dir,  f"{model}-{config['exp_name']}.pth")
        exp_value = expected_values[model]
        result = main.run(config)
        for k, v in exp_value.items():
            assert result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL), "performance of {} not correct".format(model)



# >>>>>> Test infer task of sequential models and check the shapes of predicted scores
@pytest.mark.parametrize(
        "data, models",
        [
            (
                "ml-100k",
                SEQ_MODELS
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
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        checkpoint_dir = os.path.join(config['output_path'], config['checkpoint_dir'])
        config['model_file'] = os.path.join(checkpoint_dir,  f"{model}-{config['exp_name']}.pth")
        result = main.run(config)
        shape.add(result.shape)

    assert len(shape) == 1, "shapes of scores infered by those models are different"
    


# >>>>>> Test embedding task of sequential models and check the shapes of returned embeddings
@pytest.mark.parametrize(
        "data, models",
        [
            (
                "ml-100k",
                SEQ_MODELS
            )
        ]
)
def test_item_embedding_pipeline(data, models):
    for model in models:
        config = copy.deepcopy(GLOBAL_CONF)
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['node_type'] = 'item'
        config['output_emb_file'] = f"./.temp/{data}/{model}/item_embedding/result_item_embedding.txt"
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
                SEQ_MODELS
            )
        ]
)
def test_user_embedding_pipeline(data, models):
    for model in models:
        config = copy.deepcopy(GLOBAL_CONF)
        config['dataset_path'] = os.path.join(config['dataset_path'], data)
        config['node_type'] = 'user'
        config['output_emb_file'] = f"./.temp/{data}/{model}/item_embedding/result_item_embedding.txt"
        config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}')
        checkpoint_dir = os.path.join(config['output_path'], config['checkpoint_dir'])
        config['model_file'] = os.path.join(checkpoint_dir, f"{model}-{config['exp_name']}.pth")
        user_ids, user_embs = infer_embedding.run(config)
        assert user_embs.shape[0] == len(user_ids), f"for model {model}, number of user embeddings is not equal to number of users"
        assert user_embs.shape[1] == config['embedding_size'], \
            f"for model {model}, the -1 dimension of generated user embedding is not equal to `config['embedding_size']`"



if __name__ == "__main__":
    pytest.main(["test_seq_model.py", "-s"])