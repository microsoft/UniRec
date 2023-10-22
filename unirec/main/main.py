# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

### import extranal packages here
from ast import literal_eval
import logging
import numpy as np
from typing import *
import setproctitle
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import time
import copy 
import cProfile, pstats
import io
import random
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import broadcast
### import modules defined in this project
from unirec.utils import argument_parser, logger, general
from unirec.utils.file_io import *
from unirec.facility.trainer import Trainer
from unirec.facility.solver import Solver
from unirec.constants.protocols import *

from unirec.facility.morec import (load_morec_meta_data, load_alignment_distribution, MoRecDS, 
                                   PIXController, StaticWeightSolver, PIController)

## https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)
def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def seq_collate_fn(index: Union[int, List], pad_mode:str='sequence'):
    r""" Sequence collate function.
    The function is used to collate variable-length data in batch, which is
    used in PyTorch DataLoader.
    
    Args:
        index: Union[int, List]: the column index(es) of data with variable length 
               in the batch. For list type, multiple columns corresponding to the 
               indexes need to be padded.
               E.g. for the T5 data format, the column item_seq with index 1 is padded
               in the dataloader.
    Returns:
        function: a collate function used in DataLoader.
    """
    if isinstance(index, int):
        index = [index,]
    index = set(index)
    def fn(batch):
        unzip_batch = list(zip(*batch))
        new_batch = [None] * len(unzip_batch)
        for i, d in enumerate(unzip_batch):
            if i not in index:
                if isinstance(d[0], int):
                    new_batch[i] = torch.tensor(d)
                elif isinstance(d[0], np.ndarray):
                    new_batch[i] = torch.from_numpy(np.stack(d))
            else:
                # data = [torch.from_numpy(np.copy(_t)) for _t in d]
                # new_batch[i] = pad_sequence(data, batch_first=True)
                if pad_mode == 'sequence':
                    data = general.pad_sequence_arrays(d)
                else:  # feature
                    data = general.pad_feature_arrays(d)
                new_batch[i] = torch.from_numpy(data)
        return tuple(new_batch)
    return fn


def gen_exp_name(config):
    # exp_name = uuid.uuid4()
    exp_name = config['model']
    
    if 'exp_name' in config:
        exp_name += '-' + config['exp_name']
    return exp_name

def fix_seed(config):
    if 'seed' in config:
        _seed = config['seed']
    else:
        _seed = 2022
    general.init_seed(_seed)

# def get_torch_device(config):
#     d = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     if 'device' in config:
#         d = config['device']
#     return torch.device(d)

r'''
    If existing user2history is None, load user history from file_name.
    Otherwise returns user2history directly.
    Returns:
        user2history: An N-length ndarray (N is user count) of ndarray (of history items). 
                    E.g., User2History[user_id] is an ndarray of item history for user_id. 
'''
def get_user_history(user2history, user2history_time, config, default_name):
    logger = logging.getLogger(config['exp_name'])
    file_path = config['dataset_path']
    if user2history is None:
        _user_history_filename = default_name
        _user_history_data_format = config['train_file_format']
        if 'user_history_filename' in config:
            _user_history_filename = config['user_history_filename'] 
            _user_history_data_format = config.get('user_history_file_format', _user_history_data_format)
        logger.info("Loading user history from {0} ...".format(_user_history_filename))
        user2history, user2history_time = general.load_user_history(file_path, _user_history_filename, config['n_users'], _user_history_data_format, config['time_seq'])
        logger.info("Done. {0} of users have history.".format(len(user2history)))
    return user2history, user2history_time

 
def get_data_loader(config, task, add_history_trans, MyDataSet, file_path, file_name, user2history=None, item2popularity=None, return_graph=False, item2meta=None, align_dist=None):
    config = copy.deepcopy(config)
    config['data_loader_task'] = task
    config['data_format'] = config['{0}_file_format'.format(task)] 
    config['eval_protocol'] = config.get('{0}_protocol'.format(task), None)
    num_workers = config.get('num_workers_{0}'.format(task), config['num_workers'])
    transform = None 
    if config['eval_protocol'] == EvaluationProtocal.OneVSAll.value: 
        config['n_sample_neg_{0}'.format(task)] = -1 

    if config.get('n_sample_neg_{0}'.format(task), -1) > 0:  
        transform = general.get_class_instance('AddNegSamples', 'unirec/data')(
                config['n_users'], config['n_items'], config['n_sample_neg_{0}'.format(task)],
                user2history=user2history, item_popularity=item2popularity, neg_by_pop_alpha=config.get('neg_by_pop_alpha', None)
            )
    else:
        transform = None

    if '{0}_batch_size'.format(task) in config:
        config['batch_size'] = config['{0}_batch_size'.format(task)]

    dataset = MyDataSet(config, path=file_path, filename=file_name, transform=transform)  

    if add_history_trans is not None:
        dataset.add_user_history_transform(add_history_trans)

    if return_graph:
        graph = dataset.get_graph()
        return graph

    if (task == 'train' and config['dataloader']==DatasetType.AERecDataset.value):
        index = dataset.return_key_2_index[ColNames.USER_HISTORY.value]
        if config['use_features']:
            index = [index, index+1]
        collate_fn = seq_collate_fn(index)
    elif config['data_format'] in [DataFileFormat.T5.value, DataFileFormat.T6.value]:
        _fn_col_names = [ColNames.ITEMID.value, ColNames.USER_HISTORY.value]
        _fn_idices = [dataset.return_key_2_index[k] for k in _fn_col_names if k in dataset.return_key_2_index]
        collate_fn = seq_collate_fn(_fn_idices)
    elif config['data_format'] in [DataFileFormat.T7.value]:
        _fn_col_names = [ColNames.INDEX_GROUP.value, ColNames.VALUE_GROUP.value]
        _fn_idices = [dataset.return_key_2_index[k] for k in _fn_col_names if k in dataset.return_key_2_index]
        collate_fn = seq_collate_fn(_fn_idices, pad_mode="feature")
    else:
        collate_fn = None 

    enable_morec = config.get('enable_morec', 0)
    if (enable_morec > 0) and task == 'train':
        _config = copy.deepcopy(config)
        _config['enable_morec'] = 0
        _valid_data = get_data_loader(
            _config, 'train', add_history_trans, MyDataSet, file_path, config.get('data_valid_name', 'valid'),
            user2history=user2history, item2popularity=None
            ) # the task of valid data used in out-optimization should be 'train'.
        objectives = config.get("morec_objectives", ['fairness','alignment','revenue'])
        
        morec_ds = MoRecDS(config, objectives, config['morec_ngroup'], dataset, _valid_data.dataset, 
                           alpha=config['morec_alpha'], item2meta=item2meta, align_dist=align_dist, 
                           user2history=user2history)

        data_loader = DataLoader(
            dataset=dataset, 
            pin_memory=config['pin_memory'], 
            num_workers=num_workers, 
            persistent_workers=config['persistent_workers'],
            worker_init_fn=set_worker_sharing_strategy if config['num_workers'] >= 1 else None,
            collate_fn=collate_fn,
            batch_sampler=morec_ds
        )

    else:
        data_loader = DataLoader(
            dataset=dataset, 
            batch_size=config['batch_size'], 
            shuffle=bool(config['shuffle_train']) if task == 'train' else False, 
            pin_memory=config['pin_memory'], 
            num_workers=num_workers, 
            persistent_workers=config['persistent_workers'],
            worker_init_fn=set_worker_sharing_strategy if config['num_workers'] >= 1 else None,
            collate_fn=collate_fn
        )
        
    return data_loader

def need_user_history(config):
    r"""
        On what circumstances we need the user history:
            (1) In dynamic negative sampling, we need user history to filter out false negative;
            (2) and (3): if valid or test protocol is full candidate evaluation, we need user history to skip interacted items;
    """
    if config.get('n_sample_neg_train', 0) > 0 \
        or config.get('test_protocol', None) == EvaluationProtocal.OneVSAll.value \
        or config.get('valid_protocol', None) == EvaluationProtocal.OneVSAll.value:
        return True 
    if config.get('enable_morec', 0) > 0:
        return True
    return False

def need_item_popularity(config):
    if (config.get('neg_by_pop_alpha', 0) > 0) or ("pop-kl" in config["metrics"]):
        return True
    return False


def need_item_meta_morec(config):
    if config['enable_morec'] > 0:
        return True
    else:
        if ("pop-kl" in config["metrics"]) or ("least-misery" in config["metrics"]):
            return True
    return False


def construct_item_popularity(user2history, n_items):
    res = np.zeros(n_items, dtype=np.int32)
    for v in user2history:
        if v is not None and len(v) > 0:
            if isinstance(v, np.ndarray):
                res[v] += 1
            else:
                for t in v:
                    res[t] += 1
    res[0] = 0 ## item 0 is a placeholder.
    return res

def main(config, accelerator):
    ## constants: 
    DATA_TRAIN_NAME, DATA_VALID_NAME, DATA_TEST_NAME = config.get('data_train_name', 'train'), config.get('data_valid_name', 'valid'), config.get('data_test_name', 'test')

    ## variables determined by context
    infer_score_only = False
    save_model = True
    
    ### end of declaration

    logger = logging.getLogger(config['exp_name'])
    
    dataloader_name = config['dataloader'] 
    
    file_path = config['dataset_path']
    output_path = config['output_path']   
    os.makedirs(output_path, exist_ok=True)

    ## data
    MyDataSet = general.get_class_instance(dataloader_name, 'unirec/data/dataset')      
    
    user2history = None   
    user2history_time = None 
    add_history_trans = None
    if dataloader_name in {DatasetType.SeqRecDataset.value, DatasetType.AERecDataset.value}:
        user2history, user2history_time = get_user_history(user2history, user2history_time, config, DATA_TRAIN_NAME)
        add_history_trans =  general.get_class_instance('AddUserHistory', 'unirec/data')(user2history, config['history_mask_mode'], user2history_time, config['seq_last'])
    if need_user_history(config):
        user2history, user2history_time = get_user_history(user2history, user2history_time, config, DATA_TRAIN_NAME)
    
    item2popularity = None
    if need_item_popularity(config):
        item2popularity = construct_item_popularity(user2history, config['n_items'])

    item2meta_morec = None
    alignment_distribution = None
    if need_item_meta_morec(config):
        item_meta_morec_file_path = os.path.join(config['dataset_path'], config['item_meta_morec_filename'])
        if config.get('align_dist_filename', None) is not None:
            align_dist_file_path = os.path.join(config['dataset_path'], config['align_dist_filename'])
        else:
            align_dist_file_path = None
        item2meta_morec = load_morec_meta_data(config['n_items'], item_meta_morec_file_path, config['morec_objectives'])
        alignment_distribution = load_alignment_distribution(item2meta_morec, item2popularity, align_dist_file_path)

    task = config['task']
    model_name = config['model']

    if task == TaskType.TRAIN.value:
        ## suppose the model file is placed to ./unirec/model/ and filename is the lowercase string of model name
        ## another choice for simplicity is that, you can manually import your model here
        ## from unirec.model.mf import MF   and   model = MF(config).to(config['device'])        
        model = general.get_class_instance(model_name, 'unirec/model')(config)

        if config.get('enable_morec', 0) > 0:   # load well-trained model for MoRec
            logger.info("Loading model from checkpoint: {0} ...".format(config['model_file']))
            model, cpk_config = general.load_model_freely(config['model_file'], config['device'])
            cpk_config.update(config['cmd_args']) 
            config = cpk_config

        ## prepare train data loader
        train_data = get_data_loader(
            config, 'train', add_history_trans, MyDataSet, file_path, DATA_TRAIN_NAME, 
            user2history=user2history, item2popularity=item2popularity, return_graph=not model.__optimized_by_SGD__,
            item2meta=item2meta_morec, align_dist=alignment_distribution
            )
        ## prepare valid data loader
        if model.__optimized_by_SGD__:
            valid_data = get_data_loader(
                config, 'valid', add_history_trans, MyDataSet, file_path, DATA_VALID_NAME, 
                user2history=user2history, item2popularity=item2popularity
                )
        else:
            valid_data = None
            #save_model = False  #for rec_topk, we need to save the model for inference

        logger.info(model)
    elif task in [TaskType.TEST.value, TaskType.INFER.value]: 
        model_file = config['model_file']
        logger.info("Loading model from checkpoint: {0} ...".format(model_file))
        if not (model_file.endswith(".pt") or model_file.endswith(".pth")):
            model = general.get_class_instance(model_name, 'unirec/model')(config)
            model.load(model_file)
        else:
            model, cpk_config = general.load_model_freely(model_file, config['device'])
            cpk_config.update(config['cmd_args'])
            config = cpk_config
        logger.info('Done.')
        save_model = False
    else:
        raise ValueError("Unsupported task type: {0}".format(task))

    if model.__optimized_by_SGD__:
        trainer = Trainer(config, model, accelerator)
    else:
        trainer = Solver(config, model, accelerator)
    if user2history is not None:
        trainer.set_user_history(user2history)

    if (config.get('enable_morec', 0) > 0):
        train_data.batch_sampler.set_model(trainer.model, trainer.accelerator)
        objectives = config.get("morec_objectives", ['fairness','alignment','revenue'])
        
        if config.get('morec_objective_controller', 'PID') == 'Static':    # use simple static solver
            weight = literal_eval(config['morec_objective_weights'])
            morec_oc = StaticWeightSolver(len(objectives)+1, weight)
        else:   # accuracy + at least two objectives
            if len(objectives) == 1:    # only one extra objective
                weight = [1.0]
            else:
                weight = literal_eval(config['morec_objective_weights'])
            inner_solver = StaticWeightSolver(len(objectives), weight=weight)
            morec_oc = PIXController(config['morec_expect_loss'], config['morec_beta_min'], 
                                    config['morec_beta_max'], config['morec_K_p'], config['morec_K_i'],
                                    pareto_solver=inner_solver)
        
        trainer.add_objective_controller(morec_oc)
        

    if task == TaskType.TRAIN.value:
        if valid_data:
            trainer.reset_evaluator(valid_data.dataset.config['data_format'], config['valid_protocol'])
            trainer.evaluator.set_item_meta_morec(item2meta_morec, alignment_distribution)
        try: 
            trainer.fit(
                train_data, valid_data, save_model=save_model, verbose=config['verbose'], 
                load_pretrained_model=config['load_pretrained_model'], model_file=config['model_file'] if config['load_pretrained_model'] else None
            )
        except KeyboardInterrupt:
            logger.info('Keyboard interrupt: stopping the training and start evaluating on the test set.')

   
    if task == TaskType.INFER.value:
        infer_score_only = True

    ## prepare test data loader
    test_data = get_data_loader(
        config, 'test', add_history_trans, MyDataSet, file_path, DATA_TEST_NAME, 
        user2history=user2history, item2popularity=item2popularity
        )

    trainer.reset_evaluator(test_data.dataset.config['data_format'], config['test_protocol'])
    trainer.evaluator.set_item_meta_morec(item2meta_morec, alignment_distribution)
    test_data = trainer.accelerator.prepare(test_data)
    test_result = trainer.evaluate(test_data, load_best_model=save_model, verbose=config['verbose'], predict_only=infer_score_only)

    if accelerator.is_local_main_process:
        if not infer_score_only:
            logger.info('best valid ' + f': {trainer.best_valid_result}')
            logger.info('test result' + f': {test_result}') 
            if config['use_wandb']:
                wandb_metrics = {'test/test_' + k: v for k, v in test_result.items()}
                wandb.log(wandb_metrics)

            result_file = os.path.join(output_path, 'result_{0}.{1}.{2}.tsv'.format(config['exp_name'], config['logger_time_str'], config['logger_rand']))
            logger.info('Saving test result to {0} ...'.format(result_file))
            fp = open(result_file, 'w')
            for metirc, result in test_result.items():
                fp.write(str(metirc)+'\t'+str(result)+'\n')
            fp.close()
        else:
            result_file = os.path.join(output_path, 'pred_{0}.{1}.{2}.txt'.format(config['exp_name'], config['logger_time_str'], config['logger_rand']))
            np.savetxt(result_file, test_result) # np.save(result_file, test_result)

    return test_result


def prof_to_csv(prof: cProfile.Profile):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).sort_stats('cumtime').print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = 'ncalls' + result.split('ncalls')[-1]
    lines = [','.join(line.rstrip().split(None, 5)) for line in result.split('\n')]
    return '\n'.join(lines)


def run(args: Dict=None):
    ''' Program execution entry.

    Wrapping the entry as a function could support:
        - tests pipeline scripts
        - further released package `import` operation, like "quick start". 

    Args:
        args (Dict): arguments dictionary. 

    Returns:
        Dict or numpy.ndarray: result for current task. For `train` and `test` tasks, the result would be a Dict, representing the metrics. 
                               For `infer` task, the result would be a numpy.ndarray, indicating the predicted scores.

    Note: 
        Several types of argument sources are supported now, including the `args` dictionary input here, command line arguments, 
        arguments from model checkpoint file, arguments from YAML-format config file, and default arguments in 'UniRec/unirec/config' folder. 
        The priority of those arguments sources is: `args` dict input > command line arguments > model checkpoint arguments > YAML file > default.
    '''
    # 
    # wrap the run interface in function for integration tests pipeline
    job_start_time = time.time()
    config = argument_parser.parse_arguments(args)

    if config['gpu_id']>=0: # in DDP setting, we set gpu_id to -1 and use CUDA_VISIBLE_DEVICES in script to control the gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
    accelerator = Accelerator()
    config['device'] = accelerator.device

    exp_name = gen_exp_name(config)
    config['task'] = config.get('task', 'train')
    setproctitle.setproctitle("UniRec-{0}-{1}".format(config['task'], exp_name)) 
    config['exp_name'] = exp_name
    config['cmd_args']['exp_name'] = exp_name
    logger_dir = 'output' if 'output_path' not in config else config['output_path']

    logger_tensor = torch.tensor([int(time.time()), random.randint(0,100)]).to(config['device'])
    logger_tensor = broadcast(logger_tensor, 0)
    logger_time_str = datetime.fromtimestamp(logger_tensor[0].item()).strftime("%Y-%m-%d_%H:%M:%S").replace(':', '')
    logger_rand = logger_tensor[1].item()
    config['logger_time_str'] = logger_time_str
    config['logger_rand'] = logger_rand
    config['cmd_args']['logger_time_str'] = logger_time_str
    config['cmd_args']['logger_rand'] = logger_rand
    mylog = logger.Logger(logger_dir, exp_name, logger_time_str, logger_rand, is_main_process=accelerator.is_local_main_process)

    if config['use_wandb']:
        hyper_file = os.path.join(logger_dir, '{0}.{1}.{2}.hyper_params.yaml'.format(exp_name, logger_time_str, logger_rand))
        if accelerator.is_local_main_process:
            with open(config['wandb_file'], 'r') as file:
                wandb_config = yaml.load(file, Loader=yaml.FullLoader)
            run = wandb.init(config=wandb_config)
            mylog.log(mylog.INFO, f'wandb config: {wandb.config}')
            hyper_params = {key: wandb.config[key] for key in wandb.config['parameters'].keys()}
            yaml.dump(hyper_params, open(hyper_file, 'w'))
        accelerator.wait_for_everyone()
        hyper_params = yaml.load(open(hyper_file, 'r'), Loader=yaml.FullLoader)
        mylog.log(mylog.INFO, f'hyper_params: {hyper_params},  device: {config["device"]}')
        for key in hyper_params.keys():
            config[key] = hyper_params[key]

    mylog.log(mylog.INFO, 'config='+str(config))

    fix_seed(config)
    
    pr = cProfile.Profile()
    pr.enable()    
    res = main(config, accelerator)
    pr.disable()
    
    if accelerator.is_local_main_process:
        profile_result = prof_to_csv(pr)
        profile_log_filename = mylog.filename.replace('.txt', '.prof')
        with open(profile_log_filename, 'w') as wt:
            wt.write(profile_result)

        ##=================== ending the program ======================//
        job_end_time = time.time()
        mylog.log(mylog.INFO, 'Mission complete. Time elapsed: {0:.2f} minutes.'.format((job_end_time - job_start_time)/60))
        mylog.remove_handles() 
    return res


if __name__ == '__main__': 
    run()
