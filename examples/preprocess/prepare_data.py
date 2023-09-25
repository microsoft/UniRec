# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import pickle as pkl 
import os
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import random
import yaml

from unirec.utils.file_io import *
from unirec.constants.protocols import *


def load_df_from_txt_T5_1(infile, sep=' ', has_header=False):
    userid_list = []
    item_seq_list = []
    with open(infile, 'r') as rd:
        if has_header:
            rd.readline()
        while True:
            line = rd.readline()
            if not line:
                break
            words = line[:-1].split(sep)
            userid_list.append(int(words[0]))
            item_seq_list.append(np.array([int(t) for t in words[1:]], dtype=np.int32))

    user_col_name, item_seq_col_name = ColNames.USERID.value, ColNames.USER_HISTORY.value
    return pd.DataFrame.from_dict(
        {
            user_col_name: np.array(userid_list, dtype=np.int32),
            item_seq_col_name: np.array(item_seq_list, dtype=object)
        }
    )

def load_df_from_txt_T7(infile, sep=' ', has_header=False):
    label_list = []
    index_list = []
    value_list = []
    with open(infile, 'r') as rd:
        if has_header:
            rd.readline()
        while True:
            line = rd.readline()
            if not line:
                break
            words = line[:-1].split(sep)
            label_list.append(int(words[0]))
            _index, _value = [], []
            for word in words[1:]:
                index, value = word.split(":")
                _index.append(int(index))
                _value.append(float(value))
            index_list.append(np.array(_index, dtype=np.int32))
            value_list.append(np.array(_value, dtype=np.float32))

    label_col_name, index_col_name, value_col_name = ColNames.LABEL.value, ColNames.INDEX_GROUP.value, ColNames.VALUE_GROUP.value
    return pd.DataFrame.from_dict(
        {
            label_col_name: np.array(label_list, dtype=np.int32),
            index_col_name: pd.Series(index_list),
            value_col_name: pd.Series(value_list),
            # index_col_name: np.array(index_list, dtype=object),
            # value_col_name: np.array(value_list, dtype=object),
        }
    )

r'''
The meaning of auguments:
sample_neg_k:
    do we need to sample negative?
    if sample_neg_k <= 0, don't perform negative sampling in the preprocessing stage,
    instead, sample negative items dynamically during training.
    if sample_neg_k > 0, sample k negative items for each interaction, resulting in the `one-positive in-group` data format
    i.e., for each line, user_id is an int, while item_id is a list: [I0, I1, I2, I3, ..., IK], where I0 is the positive item, I1 to Ik is sampled items.

index_by_zero:
    whether the original data file is indexed from zero
    if so, this function will add all ids by 1 to make it indexed from one.
    because UniRec suppose the position 0 is used for padding
    index_by_zero = True 
'''
def convert_to_pandas_pkl_file(
        infile, outpath, outfilename, 
        sample_neg_k=-1, index_by_zero=False, 
        names=None, header_line_cnt=0,
        data_format=None,
        sep=' ',
        save_as_behavior_sequence=False
    ):    
    # dtypes = {'user_id':int, 'item_id':int}     
    
    if data_format in {DataFileFormat.T5_1.value}:
        data = load_df_from_txt_T5_1(
            infile, sep=sep, has_header=header_line_cnt>=0
        ) 
    elif data_format in {DataFileFormat.T7.value}:
        data = load_df_from_txt_T7(
            infile, sep=sep, has_header=header_line_cnt>=0
        ) 
    else:
        data = pd.read_csv(
            infile, sep=sep, header=None if header_line_cnt<0 else header_line_cnt,
            names=names,
            engine='python'
            # dtype=dtypes
        )

    print('data shape of {0} is {1}'.format(os.path.basename(infile), data.shape))
    print('data dtypes is {0}'.format(data.dtypes))

    
     
    ### additional data transformation in dataframe
    if data_format in [DataFileFormat.T4.value]:
        data.item_id_list = data.item_id_list.apply(lambda t:np.array([int(a) for a in t.split(',')]))
        data.label_list = data.label_list.apply(lambda t:np.array([int(a) for a in t.split(',')]))

    if data_format in [DataFileFormat.T5.value]:
        data.item_seq = data.item_seq.apply(lambda t:np.array([int(a) for a in t.split(',')]))

    if data_format in [DataFileFormat.T6.value]:
        data.item_seq = data.item_seq.apply(lambda t:np.array([int(a) for a in t.split(',')]))
        data.time_seq = data.time_seq.apply(lambda t:np.array([int(a) for a in t.split(',')]))

    ## because in T5_1, the data transformation is done in data reading 
    if data_format in {DataFileFormat.T5_1.value}:
        data_format = DataFileFormat.T5.value 

    if index_by_zero:
        if data_format in [DataFileFormat.T1.value, DataFileFormat.T2.value, DataFileFormat.T2_1.value, DataFileFormat.T3.value]:
            data[['user_id', 'item_id']] += 1  
        elif data_format in [DataFileFormat.T4.value]:
            data[['user_id', 'item_id_list']] += 1  
        elif data_format in [DataFileFormat.T5.value, DataFileFormat.T6.value]:
            data[['user_id', 'item_seq']] += 1  
    
    if data_format == DataFileFormat.T2_1.value and 'session_id' not in data.columns:
        data['session_id'] = data['user_id']

    os.makedirs(outpath, exist_ok=True) 
    
    if data_format in [DataFileFormat.T7.value]:
        # max_user_id = data['sparse'].apply(lambda arr: arr[0]).max()
        max_user_id = 0
    else:
        max_user_id = data['user_id'].max()
    max_item_id = 0
    if data_format in [DataFileFormat.T1.value, DataFileFormat.T2.value, DataFileFormat.T2_1.value, DataFileFormat.T3.value]:
        max_item_id = data['item_id'].max()
    elif data_format in [DataFileFormat.T7.value]:
        # max_item_id = data['sparse'].apply(lambda arr: arr[1:].max()).max() - max_user_id
        max_item_id = 0
    elif data_format in [DataFileFormat.T4.value]:
        max_item_id = data['item_id_list'].apply(lambda t:np.max(t)).max()
    elif  data_format in [DataFileFormat.T5.value, DataFileFormat.T6.value]:
        max_item_id = data['item_seq'].apply(lambda t:np.max(t)).max()

    ## int64 is not JSON serializable
    max_user_id = int(max_user_id)
    max_item_id = int(max_item_id)

    n_lines = int(len(data))
    
    info = defaultdict(
        int, 
        {
            'n_users': max_user_id + 1, 
            'n_items': max_item_id + 1, 
            'n_lines_{0}'.format(outfilename): n_lines,
            'id_added_one': index_by_zero
        }
        ) 
    info_file = os.path.join(outpath, 'data.info') 
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r') as rd:
                pre_info = defaultdict(int, json.load(rd)) 
        except:
            pre_info = defaultdict(int)
        for key in info.keys():
            if key not in pre_info:
                pre_info[key] = info[key]
            else:
                if key.startswith('n_'):
                    pre_info[key] = max(info[key], pre_info[key])
                else:
                    if pre_info[key] != info[key]:
                        raise ValueError('key duplicated: {0}'.format(key))
        info = pre_info
    info['{0}_file_format'.format(outfilename)] = data_format

    with open(info_file, 'w') as wt:
        json.dump(info, wt)
    
    def rand_sample(x, n_item, k):
        res = np.random.randint(1, n_item, k+1)
        res[0] = x 
        return res
    def pad_zeros(x, k):
        res = np.zeros((k+1,), dtype=np.int32)
        res[0] = 1
        return res

    if save_as_behavior_sequence and data_format in [DataFileFormat.T1.value, DataFileFormat.T3.value]: 
        user_history = data.groupby('user_id')['item_id'].apply(lambda x:np.array(x)).to_frame().reset_index()
        user_history.to_feather(os.path.join(outpath, outfilename + '_as_user_history' + '.ftr'))
        save_pickle(data, os.path.join(outpath, outfilename+ '_as_user_history' + '.pkl'))

    if sample_neg_k > 0 and data_format in [DataFileFormat.T1.value, DataFileFormat.T3.value]: 
        n_item = info['n_items'] 
        if len(data) > 1000000:
            data['label'] = data['user_id'].swifter.apply(pad_zeros, k=sample_neg_k) 
            data['item_id'] = data['item_id'].swifter.apply(rand_sample, n_item=n_item, k=sample_neg_k) 
        else:
            data['label'] = data['user_id'].apply(pad_zeros, k=sample_neg_k) 
            data['item_id'] = data['item_id'].apply(rand_sample, n_item=n_item, k=sample_neg_k) 
    data = data.reset_index(drop=True)
    # data.to_csv(os.path.join(outpath, outfilename+'.tsv'),  sep='\t', index=False)
    save_pickle(data, os.path.join(outpath, outfilename+'.pkl'))
    # data.to_feather(os.path.join(outpath, outfilename+'.ftr'))
    print('In saving:')
    print(data.head(5))
    print('data.shape={0}\n'.format(data.shape))
    return info

def parse_cmd_arguments():
    parser = argparse.ArgumentParser() 
     
    parser.add_argument("--raw_datapath", type=str) 
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--outpathroot", type=str)
    parser.add_argument("--example_yaml_file", type=str)  

    parser.add_argument("--sample_negative_in_processing", type=int)   
    parser.add_argument("--index_by_zero", type=int, default=0) 
    parser.add_argument("--sep", type=str, default=' ')


    parser.add_argument("--train_file", type=str)
    parser.add_argument("--train_file_format", type=str)
    parser.add_argument("--train_file_has_header", type=int)
    parser.add_argument("--train_file_col_names", type=str)
    parser.add_argument("--train_neg_k", type=int)


    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--valid_file_format", type=str)
    parser.add_argument("--valid_file_has_header", type=int)
    parser.add_argument("--valid_file_col_names", type=str)
    parser.add_argument("--valid_neg_k", type=int)

    parser.add_argument("--test_file", type=str)
    parser.add_argument("--test_file_format", type=str)
    parser.add_argument("--test_file_has_header", type=int)
    parser.add_argument("--test_file_col_names", type=str)
    parser.add_argument("--test_neg_k", type=int)

    parser.add_argument("--full_candidate_file", type=str)
    parser.add_argument("--full_candidate_file_format", type=str)
    parser.add_argument("--full_candidate_file_has_header", type=int)
    parser.add_argument("--full_candidate_file_col_names", type=str)

    parser.add_argument("--user_history_file", type=str)
    parser.add_argument("--user_history_file_format", type=str)
    parser.add_argument("--user_history_file_has_header", type=int)
    parser.add_argument("--user_history_file_col_names", type=str) 

    parser.add_argument("--group_size", type=int) 
        
        
    (args, unknown) = parser.parse_known_args()  
    # print(args)
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None and value not in ['none', 'None']:
            parsed_results[arg] = value
    

    return parsed_results
 

def process_transaction_dataset(arguments): 
    raw_datapath = arguments['raw_datapath']
    dataset_name = arguments['dataset_name']
    dataset_outpathroot = arguments['outpathroot']
    example_yaml_file = arguments['example_yaml_file']
    

    index_by_zero = arguments.get('index_by_zero', 0)
    sep = arguments.get('sep', '\t')  
    np.set_printoptions(linewidth=np.Inf)


    train_neg_k = arguments.get('train_neg_k', 0)  
    valid_neg_k = arguments.get('valid_neg_k', 0)  
    test_neg_k = arguments.get('test_neg_k', 0)

    group_size = arguments.get('group_size', -1)

    for data_name in ['train', 'valid', 'test', 'full_candidate', 'user_history']:
        if '{0}_file'.format(data_name) not in arguments:
            continue
        header_line_cnt = arguments.get('{0}_file_has_header'.format(data_name), 0) - 1  ## -1 means no header; 0 means has header
        data_format = arguments.get('{0}_file_format'.format(data_name), DataFileFormat.T1.value)    
        info = convert_to_pandas_pkl_file(
            os.path.join(raw_datapath, arguments['{0}_file'.format(data_name)]),
            os.path.join(dataset_outpathroot, dataset_name),  
            data_name, 
            sample_neg_k=arguments.get('{0}_neg_k'.format(data_name), 0), 
            index_by_zero=index_by_zero,
            names=eval(arguments.get('{0}_file_col_names'.format(data_name), "['user_id', 'item_id']")),
            header_line_cnt=header_line_cnt,
            data_format=data_format,
            sep=sep,
            save_as_behavior_sequence=False
        )
        
    
    with open(example_yaml_file, 'r') as rd:
        data_config = yaml.safe_load(rd)
    data_config['n_users'] = info['n_users']
    data_config['n_items'] = info['n_items'] 
    if arguments.get('train_file_format'.format(data_name), DataFileFormat.T1.value) in [DataFileFormat.T7.value]:
        rawdata_info_path = os.path.join(raw_datapath, 'raw_data.info')
        if not os.path.exists(rawdata_info_path):
            assert AttributeError('You need to provide data details in raw data directory.')
        with open(rawdata_info_path, 'r') as rd:
            rawdata_info = defaultdict(int, json.load(rd))
        if 'n_feats' not in rawdata_info:
            assert AttributeError("You need to provide both n_feats (the number of features).")
        data_config['n_feats'] = rawdata_info['n_feats']
    data_config['n_neg_train_from_sampling'] = train_neg_k
    data_config['n_neg_valid_from_sampling'] = valid_neg_k
    data_config['n_neg_test_from_sampling'] = test_neg_k
    data_config['group_size'] = group_size
    for k,v in info.items():
        if '_file_format' in k:
            data_config[k] = v

    
    dataset_yaml_file = os.path.join(
        os.path.dirname(example_yaml_file), 
        dataset_name+'.yaml'
    )
    with open(dataset_yaml_file, 'w') as wt:
        yaml.dump(data_config, wt, default_flow_style=False)


r'''
    Convert text data files into pandas dataframe and save as binary files.
    Input files:
        train_file, valid_file, test_file, example_yaml_file
        user_history_file
    Suggested format for instance files:
        userId,movieId(other optinal columns, such as rating,timestamp)
    If index_by_zero, user_id and item_id will add 1 to make them indexed starting from 1.
    If sample_negative_in_processing=1, random negatives will be added to the train/valid/test files.

    Notes:        
        Must provide col_names for each data file. For some overlap basic columns, please use the names in constants.protocals.ColNames
'''
if __name__ == '__main__':  
    arguments = parse_cmd_arguments()
    print(arguments)
    process_transaction_dataset(arguments)
