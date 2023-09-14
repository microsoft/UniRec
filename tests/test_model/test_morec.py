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
    'user_pre_item_emb': 0,
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
    'metrics': "['hit@5;10', 'ndcg@5;10', 'rhit@5;10', 'rndcg@5;10']",
    'key_metric': "ndcg@5",
    'num_workers': 4,
    'num_workers_test': 0,
    'verbose': 2,
    'neg_by_pop_alpha': 0,
    'item_price_filename': 'item_price.csv',
    'item_category_filename': 'item_category.csv',
}


EXPECTED_METRICS = {
                        "Pretrain": {"hit@5": 0.06815, "ndcg@5": 0.04292},
                        "Finetune": {"hit@5": 0.07135, "ndcg@5": 0.04829}
                    }

MODEL2DATALOADER = {
    "MF": "BaseDataset",
    "MultiVAE": "AERecDataset", 
}

# >>>>>> Test train pipeline of MoRec with MF and check the performance
@pytest.mark.parametrize(
        "data, model, expected_values",
        [
            (
                "ml-100k",
                'MF',
                EXPECTED_METRICS['Pretrain']
            )
        ]
)
def test_morec_pretrain(data, model, expected_values):
    # pretrain
    pretrain_config = copy.deepcopy(GLOBAL_CONF)
    pretrain_config['dataset'] = data
    pretrain_config['dataset_path'] = os.path.join(pretrain_config['dataset_path'], data)
    pretrain_config['model'] = model
    pretrain_config['enable_morec'] = 0
    pretrain_config['checkpoint_dir'] = "morec_pretrain_" + pretrain_config['checkpoint_dir']
    pretrain_config['exp_name'] = 'pytest_morec_pretrain'
    pretrain_config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}/morec')
    pretrain_config['dataloader'] = MODEL2DATALOADER[model]
    result = main.run(pretrain_config)
    for k, v in expected_values.items():
        assert result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL), "value of {} not correct".format(k)




@pytest.mark.parametrize(
        "data, model, expected_values",
        [
            (
                "ml-100k",
                'MF',
                EXPECTED_METRICS['Finetune']
            )
        ]
)
def test_morec_finetune(data, model, expected_values):
    config = copy.deepcopy(GLOBAL_CONF)
    config['dataset'] = data
    config['dataset_path'] = os.path.join(config['dataset_path'], data)
    config['model'] = model
    config['dataloader'] = MODEL2DATALOADER[model]
    config['epochs'] = 2
    config['enable_morec'] = 1
    config['exp_name'] = 'pytest_morec_finetune'
    config['output_path'] = os.path.join(UNIREC_PATH, f'tests/.temp/output/{data}/{model}/morec')
    config['model_file'] = os.path.join(config['output_path'], "morec_pretrain_"+config['checkpoint_dir'], f"{model}-pytest_morec_pretrain.pth")
    config['checkpoint_dir'] = "morec_finetune_" + config['checkpoint_dir']

    # MoRec parameters
    config['morec_objectives']=['fairness', 'alignment', 'revenue']
    config["morec_ngroup"] = 5
    config["morec_alpha"] = 0.01
    config["morec_expect_loss"] = 0.25
    config["morec_beta_min"] = 0.1
    config["morec_beta_max"] = 1.5
    config["morec_K_p"] = 0.05
    config["morec_K_i"] = 0.001
    config["morec_objective_weights"] = "[0.1,0.1,0.8]"
    config['metrics'] = "['hit@5', 'ndcg@5', 'rhit@5', 'rndcg@5', 'pop-kl@5', 'least-misery']"
    result = main.run(config)
    for k, v in expected_values.items():
        assert result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL), "value of {} not correct".format(k)

if __name__ == "__main__":
    pytest.main(["test_morec.py", "-s"])
