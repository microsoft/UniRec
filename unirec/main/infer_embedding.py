# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import random
import inspect
import logging
import argparse
import numpy as np
import setproctitle
from typing import *
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
### import modules defined in this project
from unirec.constants.protocols import *
from unirec.data.dataset import inferdataset
from unirec.utils import logger, general, argument_parser

@torch.no_grad()
def infer_embedding(test_data_loader, model, config, accelerator):
    model.eval()
    model = accelerator.unwrap_model(model)
    iter_data = (
        tqdm(
            enumerate(test_data_loader),
            total=len(test_data_loader),
            desc="infer embeddings",
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process
        )
    )
    all_embeddings = []
    all_ids = []
    for batch_idx, inter_data in iter_data:
        samples = {k:inter_data[v] for k,v in test_data_loader.dataset.return_key_2_index.items()}
        if config['node_type'] == 'user':
            inputs = {k: v for k, v in samples.items() if k in inspect.signature(model.forward_user_emb).parameters}
            embeddings = model.forward_user_emb(**inputs) if model.__optimized_by_SGD__ else np.array(model.forward_user_emb(**inputs).toarray())
            embeddings = accelerator.gather_for_metrics(torch.as_tensor(embeddings, device=config['device']).contiguous()).cpu().numpy()
            data_ids = accelerator.gather_for_metrics(samples['user_id']).cpu().numpy()
        elif config['node_type'] == 'item':
            embeddings = model.forward_item_emb(**samples) if model.__optimized_by_SGD__ else np.array(model.forward_item_emb(**samples).toarray())
            embeddings = accelerator.gather_for_metrics(torch.as_tensor(embeddings, device=config['device']).contiguous()).cpu().numpy()
            data_ids = accelerator.gather_for_metrics(samples['items']).cpu().numpy()
        else:
            raise NotImplementedError('node_type {0} is not supported'.format(config['node_type']))
        all_embeddings.append(embeddings)
        all_ids.append(data_ids)
    all_embeddings = np.vstack(all_embeddings)
    all_ids = np.hstack(all_ids)
    return all_embeddings, all_ids

   
def load_and_infer(config, accelerator) -> Tuple[np.ndarray]:
    logger = logging.getLogger(config['exp_name']) 
    logger.info(str(config))
    
    model_path = config['model_file']
    logger.info('Loading model from {0}'.format(model_path))
    model, cpk_config = general.load_model_freely(model_path, config['device'])
    cpk_config.update(config)
    config = cpk_config 

    if 'test_batch_size' in config:
        config['batch_size'] = config['test_batch_size']

    dataset_path = config['dataset_path']
    if 'id_file_name' in config:
        test_ids = np.loadtxt(os.path.join(dataset_path, config['id_file_name']), dtype=np.int64).reshape(-1)
    else:
        test_ids = np.arange(config['n_users'], dtype=np.int64) if config['node_type']=='user' else np.arange(config['n_items'], dtype=np.int64)
    logger.info('#. {0}s for inference: {1}'.format(config['node_type'], len(test_ids))) 
    
    logger.info('loading user history...')
    user_history_filename = config['user_history_filename']
    user_history_format = config.get('user_history_file_format', None)
    user_history, user_history_time = general.load_user_history(dataset_path, user_history_filename, format=user_history_format)   
    logger.info('finished loading user history.') 
  
    logger.info('loading dataset...')
    is_seqrec = True if (('SeqRecBase' in model.annotations) or ('AERecBase' in model.annotations)) and config['node_type']=='user' else False
    test_data = inferdataset.InferDataset(config, test_ids, user_history, is_seqrec)
    
    test_data_loader = DataLoader(
        dataset=test_data, 
        batch_size=config['batch_size'], 
        shuffle=False,  
        num_workers=0, 
    )

    model, test_data_loader = accelerator.prepare(model, test_data_loader) 
    embeddings, ids = infer_embedding(test_data_loader, model, config, accelerator)
    
    outpath = config['output_emb_file']
    logger.info('saving inferred embeddings to {0}'.format(outpath))
    if accelerator.is_local_main_process:
        with open(outpath, 'w') as wt:
            for id, embedding in zip(ids, embeddings):
                wt.write('{0}\t{1}\n'.format(id, ','.join([str(x) for x in embedding]))) 
    return ids, embeddings
  
 
def parse_cmd_arguments():
    parser = argparse.ArgumentParser()
      
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--id_file_name", type=str, help='each line contains a user/item id, if not specified, all users/items will be used for inference')
    parser.add_argument("--user_history_filename", type=str)
    parser.add_argument("--user_history_file_format", type=str)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--test_batch_size", type=int)  
    parser.add_argument("--output_emb_file", type=str)
    parser.add_argument("--node_type", type=str, default='user', choices=['user', 'item'])
    parser.add_argument("--last_item", type=int, default=0, help='indicate the position of the target item in the user history, \
                            0 means use all history to encode user embedding; otherwise use history[:-last_item] to encode user embedding')
    parser.add_argument("--features_filepath", type=str)

    (args, unknown) = parser.parse_known_args()  
    # print(args)
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None and value not in ['none', 'None']:
            parsed_results[arg] = value
    
    return parsed_results



def run(args: Dict=None) -> Tuple[np.ndarray]:
    # `args` is used in test scripts
    # priority: args > cmd args (user-given cmd args > default cmd args)
    exp_name = 'infer_embedding'
    config = {}
    config['exp_name'] = exp_name
    cmd_config = parse_cmd_arguments()
    config.update(cmd_config)
    if args is not None:
        config.update(args)

    setproctitle.setproctitle(config['exp_name'])  
    logger_dir = 'output' if 'output_emb_file' not in config else os.path.dirname(config['output_emb_file'])
    logger_time_str = general.get_local_time_str().replace(':', '')
    logger_rand = random.randint(0, 100)
    
    accelerator = Accelerator()
    config['device'] = accelerator.device
    
    config['logger_time_str']=logger_time_str
    config['logger_rand']=logger_rand
    mylog = logger.Logger(logger_dir, exp_name, time_str=logger_time_str, rand=logger_rand, is_main_process=accelerator.is_local_main_process) 
    ids, embeddings = load_and_infer(config, accelerator)
    return ids, embeddings


if __name__ == '__main__':
    run()
