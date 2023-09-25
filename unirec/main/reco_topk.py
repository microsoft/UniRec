# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import setproctitle
import logging
import copy
import argparse
from tqdm import tqdm
import random
import inspect
from accelerate import Accelerator
### import modules defined in this project
from unirec.utils import logger, general
from unirec.data.dataset import inferdataset
from unirec.constants.protocols import *

@torch.no_grad()
def _get_tok_recommendations(test_data_loader, model, user_history, topk, test_config, accelerator):
    if test_config['item_file'] and accelerator.is_local_main_process:
        user2itemids = {} #change to dict to notify the user id
        with open(test_config['item_file'], 'r') as f:
            for line in f:
                user, items = line.strip().split('\t')
                user2itemids[int(user)] = [int(i) for i in items.split(',')]
        outfile=open(test_config['output_path'], 'w')
    model.eval()
    model = accelerator.unwrap_model(model)
    n_user = len(test_data_loader.dataset.data_ids)
    res_reco = np.zeros((n_user, topk), dtype=np.int32) 
    item_embeddings = model.forward_all_item_emb()
    if model.has_item_bias:
        item_bias = model.get_all_item_bias()
        item_bias = item_bias.reshape((1, -1))
    
    iter_data = (
        tqdm(
            enumerate(test_data_loader),
            total=len(test_data_loader),
            desc="TopK inference",
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process
        )
    )

    start_idx = 0
    for batch_idx, inter_data in iter_data:
        samples = {k:inter_data[v] for k,v in test_data_loader.dataset.return_key_2_index.items()}
        inputs = {k: v for k, v in samples.items() if k in inspect.signature(model.forward_user_emb).parameters}
        user_embeddings = model.forward_user_emb(**inputs).detach().cpu().numpy() if model.__optimized_by_SGD__ else model.forward_user_emb(**inputs)
        # samples['user_id'] = samples['user_id'].cpu().numpy() if isinstance(samples['user_id'], torch.Tensor) else samples['user_id']
        
        batch_scores = np.matmul(user_embeddings, item_embeddings.T) if model.__optimized_by_SGD__ else np.array(model.sparse_matrix_mul(user_embeddings, item_embeddings))

        if model.has_item_bias:
            batch_scores += item_bias
        if model.has_user_bias:
            user_bias = model.get_user_bias(samples)
            batch_scores += user_bias.reshape(-1, 1)
        batch_scores = batch_scores / test_config['tau']
        
        user_ids = accelerator.gather_for_metrics(samples['user_id']).cpu().numpy()
        if test_config['item_file']:
            batch_scores = accelerator.gather_for_metrics(torch.tensor(batch_scores, device=accelerator.device)).cpu().numpy()
            if accelerator.is_local_main_process:
                for idx, userid in enumerate(user_ids):
                    target = user_history[userid][-test_config['last_item']] if test_config['last_item']>0 else -99
                    for item in user2itemids[userid]:
                        label = '1' if int(item) == target else '0'
                        score = batch_scores[idx][int(item)] if int(item)>0 else 0.0
                        outfile.write(str(userid)+'\t'+str(item)+'\t'+str(score)+ '\t' + label + '\n')
        
        else:
            for idx, userid in enumerate(samples['user_id'].cpu().numpy()):
                if userid < len(user_history) and user_history[userid] is not None:
                    history = user_history[userid]
                    if test_config['last_item']>0:
                        target_item = history[-test_config['last_item']]
                        target_score = batch_scores[idx][target_item]
                    batch_scores[idx][history] = -np.Inf
                batch_scores[idx][0] = -np.Inf
                if test_config['last_item']>0:
                    batch_scores[idx][target_item] = target_score

            sorted_index = general.get_topk_index(batch_scores, topk)
            sorted_index = accelerator.gather_for_metrics(torch.tensor(sorted_index, device=accelerator.device)).cpu().numpy()
            end_idx = start_idx + len(user_ids)
            res_reco[start_idx:end_idx,:] = sorted_index
            start_idx = end_idx
    if test_config['item_file'] and accelerator.is_local_main_process:
        outfile.close()
    return res_reco

   
def do_topk_reco(config, accelerator):
    logger = logging.getLogger(config['exp_name']) 
    logger.info(str(config))
    
    model_path = config['model_file']
    test_data_path = config['dataset_path']
    test_data_name = config['dataset_name'] ## should be a user id file

    user_history_2_mask = config['user_history_filename']
    user_history_format = config.get('user_history_file_format', None)
    outpath = config['output_path']
    
    logger.info('Loading model from {0}'.format(model_path))
    model, cpk_config = general.load_model_freely(model_path, config['device'])
    cpk_config.update(config)
    config = cpk_config 
    test_config = copy.deepcopy(config)

    if 'test_batch_size' in test_config:
        test_config['batch_size'] = test_config['test_batch_size']
        
    test_userids = np.loadtxt(os.path.join(test_data_path, test_data_name), dtype=np.int64).reshape(-1) 
    logger.info('#. users for recommendations: {0}'.format(len(test_userids)))
    
    logger.info('loading user history...')
    user_history, user_history_time = general.load_user_history(test_data_path, user_history_2_mask, format=user_history_format)   
    logger.info('finished loading user history.') 
  
    logger.info('loading dataset...')
    is_seqrec = False if 'SeqRecBase' not in model.annotations else True
    test_data = inferdataset.InferDataset(test_config, test_userids, user_history, is_seqrec)
    
    test_data_loader = DataLoader(
        dataset=test_data, 
        batch_size=test_config['batch_size'], 
        shuffle=False,  
        num_workers=0, 
    )

    model, test_data_loader = accelerator.prepare(model, test_data_loader) 
    reco_list = _get_tok_recommendations(test_data_loader, model, user_history, test_config['topk'], test_config, accelerator)
    
    if not test_config['item_file'] and accelerator.is_local_main_process:
        logger.info('saving topk recommendations to {0}'.format(outpath))
        np.savetxt(outpath, reco_list, delimiter=',', fmt='%i') 
  
 
def parse_cmd_arguments():
    parser = argparse.ArgumentParser()
      
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model_file", type=str)   ## rename from model_path to model_file
    parser.add_argument("--test_batch_size", type=int)  
    parser.add_argument("--user_history_filename", type=str)  ## rename from user_history_2_mask to user_history_filename
    parser.add_argument("--user_history_file_format", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--last_item", type=int, default=0, help='indicate the position of the target item in the user history, \
                            0 means use all history to encode user embedding; otherwise use history[:-last_item] to encode user embedding')
    parser.add_argument("--topk", type=int, default=100, help='topk for recommendation')
    parser.add_argument("--item_file", type=str, default='', help='item file for getting scores')
    parser.add_argument("--features_filepath", type=str)
        
    (args, unknown) = parser.parse_known_args()  
    # print(args)
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None and value not in ['none', 'None']:
            parsed_results[arg] = value
    
    return parsed_results


if __name__ == '__main__':
    config = parse_cmd_arguments()
    exp_name = 'reco_topk'
    setproctitle.setproctitle(exp_name)  
    config['exp_name'] = exp_name
    logger_dir = 'output' if 'output_path' not in config else os.path.dirname(config['output_path'])
    logger_time_str = general.get_local_time_str().replace(':', '')
    logger_rand = random.randint(0, 100)

    accelerator = Accelerator()
    config['device'] = accelerator.device

    config['logger_time_str']=logger_time_str
    config['logger_rand']=logger_rand
    mylog = logger.Logger(logger_dir, exp_name, time_str=logger_time_str, rand=logger_rand, is_main_process=accelerator.is_local_main_process) 
    do_topk_reco(config, accelerator)
