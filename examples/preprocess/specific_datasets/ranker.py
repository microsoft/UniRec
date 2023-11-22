
import argparse
import json
import os
import random

import pandas as pd
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from copy import deepcopy


r'''
This function preprocesses the single column user history file and generates sequential user history file.
'''
def data_formatting(prefile, outfile, sep: str=",", input_file_format: str="user-item"):
    assert input_file_format in {"user-item", "user-item_seq"}, "`input_file_format` must be `user-item` or `user-item_seq`"
    if input_file_format == "user-item":
        df = pd.read_csv(prefile, header=0, names=['user_id', 'item_id'], dtype={'user_id': int, 'item_id': str}, sep=sep, engine='python')

        df['item_id'] = df['item_id'].apply(lambda x: [x])
        df_grouped = df.groupby('user_id').agg({'item_id': lambda x: [item[0] for item in x.drop_duplicates()]})
    else:
        df = pd.read_csv(prefile, header=0, names=['user_id', 'item_seq'], dtype={'user_id': int, 'item_seq': str}, sep=sep, engine='python')
        df['item_id'] = df['item_seq'].apply(lambda x: x.split(","))
        df_grouped = df[['user_id', 'item_id']]
    user_items_dict = df_grouped.set_index('user_id')['item_id'].to_dict()
    parent_dir = os.path.dirname(outfile)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    with open(outfile, 'w') as wt:
        for userid, items in user_items_dict.items():
            line = f"{userid} {' '.join(map(str, items))}\n"
            wt.write(line)


r'''
This function randomly samples n_neg_sample items from full_item_list.
'''
def random_neg_sample(full_item_list, n_neg_sample, item_to_remove):
    temp_item_list = full_item_list.copy()
    for item in item_to_remove:
        if item in temp_item_list:
            temp_item_list.remove(item)
    neg_item_list = random.sample(temp_item_list, n_neg_sample)
    return neg_item_list


r'''
This function write the dict data pair into file.
'''
def write_dict_to_file(d, outfile):
    wt = open(outfile, 'w')
    for k, v in d.items():
        wt.write(f"{k} {v}\n")
    wt.close()


r'''
This function preprocesses the user history file and generates various output files for training, validation, and testing in libfm data format.

Args:
args: An argparse.Namespace object containing the following attributes:
    - infile (str): The input file containing the raw data.
    - outdir (str): The directory to store the output files.
    - n_neg_k (int): The number of negative samples for each positive sample (default is 5).

Outputs:
The function generates the following files in the output directory:
    - train.txt
    - valid.txt
    - test.txt
    - user_history.txt
    - user2uid.txt
    - item2tid.txt
    - item2hid.txt
    - raw_data.info

Example:
    Input data format (user, item1, item2, ..., itemN):
    1 101 102 103
    2 201 202 203 204
    3 301 302 303 304 305

    Input summary:
        n_users         : 3
        n_target_items  : 12
        n_hist_items    : 9

    Output train samples (label, user, item, history):
        1 1:1 4:1
        0 1:1 11:1
        ...
        1 2:1 8:1 18:1
        0 2:1 4:1 18:1
        ...
    Output valid samples (label, user, item, history):
        1 1:1 5:1 16:1
        0 1:1 8:1 16:1
        ...  
    Output test samples (label, user, item, history):
        1 1:1 6:1 16:1 17:1
        0 1:1 9:1 16:1 17:1
        ...
'''
def run_libfm(arguments):
    wt_train = open(arguments['train_file'], 'w')
    wt_valid = open(arguments['valid_file'], 'w')
    wt_test = open(arguments['test_file'], 'w')
    wt_user_history = open(arguments['user_history_file'], 'w')

    pos_user_history = dict()
    history_item_set = set()
    target_item_set = set()
    lengths = []
    train_pos_items = []
    valid_pos_items = []
    test_pos_items = []
    _cnt = 0
    with open(arguments['infile'], 'r') as rd:
        for line in rd.readlines():
            words = line.strip().split(' ')
            user, items = words[0], words[1:]
            item_set = set()
            dedup_items = []
            for item in items:
                if item not in item_set:
                    item_set.add(item)
                    dedup_items.append(item)
            items = dedup_items
            _cnt += len(items)
            item_len = len(items)
            if item_len < 3:
                continue
            lengths.append(item_len)
            train_pos_items.append(items[:-2])  # flatten the train items
            valid_pos_items.append(items[-2])
            test_pos_items.append(items[-1])

            pos_user_history[user] = items
            target_item_set = target_item_set | set(items)
            history_item_set = history_item_set | set(items[:-1])
    print(f"#. intersections: {_cnt}.")

    sorted_user_list = sorted(set(pos_user_history.keys()), key=int)
    n_users = len(sorted_user_list)
    target_item_list = sorted(target_item_set, key=int)
    n_target_items = len(target_item_list)
    history_item_list = sorted(history_item_set, key=int)
    n_hist_items = len(history_item_list)

    # plus 1 for history zero padding
    user2uid = {user: uid for user, uid in zip(sorted_user_list, range(1, n_users+1))}
    item2tid = {item: tid+n_users+1 for item, tid in zip(target_item_list, range(n_target_items))}
    item2hid = {item: hid+n_users+n_target_items+1 for item, hid in zip(history_item_list, range(n_hist_items))}

    # skip userid
    # item2tid = {item: tid for item, tid in zip(target_item_list, range(1, n_target_items+1))}
    # item2hid = {item: hid+n_target_items+1 for item, hid in zip(history_item_list, range(n_hist_items))}

    write_dict_to_file(user2uid, arguments['user2uid_file'])
    write_dict_to_file(item2tid, arguments['item2tid_file'])
    write_dict_to_file(item2hid, arguments['item2hid_file'])

    def write_to_file(wt, user, items, labels, hist):
        # if not hist:
        #     return
        uid = user2uid[user]
        tids = [item2tid[item] for item in items]
        hids = [item2hid[item] for item in hist]
        # len_seq = len(hids)
        # res = [0] * k
        # if len_seq < k:
        #     res[(k-len_seq):] = hids[:]
        # else:
        #     res[:] = hids[len_seq-k:]
        hist_str = " ".join([f"{hid}:1" for hid in hids])
        for label, tid in zip(labels, tids):
            data_str = f"{label} {uid}:1 {tid}:1"
            data_str += f" {hist_str}\n" if hist_str else "\n"
            wt.write(data_str)

            # skip userid
            # wt.write(f"{label} {tid}:1 {hist_str}\n")

    n_neg_k = arguments['n_neg_k']
    labels = ['1'] + ['0'] * n_neg_k
    for (user, hist), itrains, ivalid, itest in zip(pos_user_history.items(), train_pos_items, valid_pos_items, test_pos_items):
        for i, itrain in enumerate(itrains):
            train_items = [itrain] + random_neg_sample(target_item_list, n_neg_k, hist)
            write_to_file(wt_train, user, train_items, labels, hist[:i])

        valid_items = [ivalid] + random_neg_sample(target_item_list, n_neg_k, hist)
        test_items = [itest] + random_neg_sample(target_item_list, n_neg_k, hist)

        write_to_file(wt_valid, user, valid_items, labels, hist[:-2])
        write_to_file(wt_test, user, test_items, labels, hist[:-1])

        wt_user_history.write(f"{user} {','.join(hist)}\n")

    wt_train.close()
    wt_valid.close()
    wt_test.close()
    wt_user_history.close()
    
    n_feats = n_users+n_target_items+n_hist_items+1

    # skip userid
    # n_feats = n_target_items+n_hist_items+1
    rawdata_config = {
        'n_feats': n_feats,
    }
    with open(arguments['rawdata_info_file'], 'w') as wt:
        json.dump(rawdata_config, wt)

    print(f'max item length: {max(lengths)}')
    print(f'min item length: {min(lengths)}')
    print(f'#. features: {n_feats}')


r'''
This function preprocesses the user history file and generates various output files for training, validation, and testing in T4 data format.

Args:
args: An argparse.Namespace object containing the following attributes:
    - infile (str): The input file containing the raw data.
    - outdir (str): The directory to store the output files.
    - n_neg_k (int): The number of negative samples for each positive sample (default is 5).

Outputs:
The function generates the following files in the output directory:
    - train.txt
    - valid.txt
    - test.txt
    - user_history.txt
    - user2uid.txt
    - item2tid.txt

Example:
    Input data format (user, item1, item2, ..., itemN):
    1 101 102 103
    2 201 202 203 204
    3 301 302 303 304 305

    Input summary:
        n_users  : 3
        n_items  : 12

    Output train samples (user, item_list, label_list):
        1 1,8,12,10,4,6 1,0,0,0,0,0
        2 4,2,9,11,1,10 1,0,0,0,0,0
        2 5,3,1,10,12,2 1,0,0,0,0,0
        ...
    Output valid samples (user, item_list, label_list):
        1 2,4,7,10,11,8 1,0,0,0,0,0
        ...  
    Output test samples (user, item_list, label_list):
        1 3,6,12,11,4,9 1,0,0,0,0,0
        ...
'''
def run_rank(arguments):
    wt_train = open(arguments['train_file'], 'w')
    wt_valid = open(arguments['valid_file'], 'w')
    wt_test = open(arguments['test_file'], 'w')
    wt_user_history = open(arguments['user_history_file'], 'w')

    pos_user_history = dict()
    all_item_set = set()
    lengths = []
    train_pos_items = []
    valid_pos_items = []
    test_pos_items = []
    _cnt = 0
    with open(arguments['infile'], 'r') as rd:
        for line in rd.readlines():
            words = line.strip().split(' ')
            user, items = words[0], words[1:]
            item_set = set()
            dedup_items = []
            for item in items:
                if item not in item_set:
                    item_set.add(item)
                    dedup_items.append(item)
            items = dedup_items
            _cnt += len(items)
            item_len = len(items)
            if item_len < 3:
                continue
            lengths.append(item_len)
            train_pos_items.append(items[:-2])  # flatten the train items
            valid_pos_items.append(items[-2])
            test_pos_items.append(items[-1])

            pos_user_history[user] = items
            all_item_set = all_item_set | item_set
    print(f"#. intersections: {_cnt}.")

    sorted_user_list = sorted(set(pos_user_history.keys()), key=int)
    n_users = len(sorted_user_list)
    all_item_list = sorted(all_item_set, key=int)
    n_items = len(all_item_list)

    # plus 1 for history zero padding
    user2uid = {user: uid for user, uid in zip(sorted_user_list, range(1, n_users+1))}
    item2tid = {item: tid for item, tid in zip(all_item_list, range(1, n_items+1))}

    write_dict_to_file(user2uid, arguments['user2uid_file'])
    write_dict_to_file(item2tid, arguments['item2tid_file'])

    def write_to_file(wt, user, items, labels=None):
        uid = user2uid[user]
        tids = [str(item2tid[item]) for item in items]
        data_str = f"{uid} {','.join(tids)}"
        data_str += f" {','.join(labels)}\n" if labels is not None else "\n"
        wt.write(data_str)

    n_neg_k = arguments['n_neg_k']
    labels = ['1'] + ['0'] * n_neg_k
    for (user, hist), itrains, ivalid, itest in zip(pos_user_history.items(), train_pos_items, valid_pos_items, test_pos_items):
        for itrain in itrains:
            train_items = [itrain] + random_neg_sample(all_item_list, n_neg_k, hist)
            write_to_file(wt_train, user, train_items, labels)

        valid_items = [ivalid] + random_neg_sample(all_item_list, n_neg_k, hist)
        test_items = [itest] + random_neg_sample(all_item_list, n_neg_k, hist)

        write_to_file(wt_valid, user, valid_items, labels)
        write_to_file(wt_test, user, test_items, labels)

        write_to_file(wt_user_history, user, hist[:-1])

    wt_train.close()
    wt_valid.close()
    wt_test.close()
    wt_user_history.close()

    print(f'max item length: {max(lengths)}')
    print(f'min item length: {min(lengths)}')


"""
This function samples a specified number of items from a list of candidate items, excluding the target item and items already in the sequence.
"""
def sample_items_from_candidates(candi_items, sampled_items_num, target_item, item_seq):
    unique_candi_items = None
    max_retry_cnt = 100

    cur_sampled_items = []
    for _ in range(sampled_items_num):
        _cnt = 0
        sampled_flag = False
        while _cnt < max_retry_cnt:
            sampled_item = random.sample(candi_items, 1)[0]
            _cnt += 1
            sampled_flag = not ((sampled_item == target_item) or (sampled_item in item_seq) or (sampled_item in cur_sampled_items))
            if sampled_flag:
                break
        if _cnt == max_retry_cnt and not sampled_flag:
            if unique_candi_items is None:
                unique_candi_items = set(candi_items) - set([target_item] + item_seq)
            else:
                unique_candi_items = _unique_candi_items
            _unique_candi_items = unique_candi_items - set(cur_sampled_items)
            if len(_unique_candi_items) == 0:
                raise ValueError("No candidate items to sample from, you may need to decrease the number of negative samples.")
            _candi_items = [item for item in candi_items if item in _unique_candi_items]
            sampled_item = random.sample(_candi_items, 1)[0]
        cur_sampled_items.append(sampled_item)

    return cur_sampled_items


r'''
This function implements a negative sampling strategy for generating negative samples for Ada-Ranker.

The function operates as follows:
    1. Determines the categories to sample from, including the positive item's category and up to two additional random categories.
    2. Determines the number of samples to draw from each category using a multinomial distribution with equal probabilities for each category.
    3. For each category to draw samples from, it decides randomly whether to draw from a uniform distribution of items or a popularity-biased distribution.
    4. Draws the specified number of samples from the chosen category and distribution, excluding the positive item and any other items specified in item_to_remove.
'''
def distritbuion_mixer_sampling(cate, tgt_item, cate_num, cate2item, cate2item_uni, n_neg_sample, item_to_remove):
    sampled_cate = [cate]
    multicate_num = random.sample([0, 1, 2], 1)[0]
    sampled_cate += random.sample(range(1, cate_num+1), multicate_num)

    sample_ratio = np.ones(len(sampled_cate)) / len(sampled_cate)
    sample_num = np.random.multinomial(n_neg_sample, sample_ratio, size=1)[0]

    sampled_items = []

    seed = random.sample(range(0, 100), 1)[0]
    for idx in range(len(sample_num)):
        sampled_items_num = sample_num[idx]
        if sampled_items_num == 0:
            continue
        cate_ = sampled_cate[idx]
        # 50% universal 50% pop
        if seed < 50:  # universal
            candi_items = cate2item_uni[cate_]
        else:  # pop
            candi_items = cate2item[cate_]

        cur_sampled_items = sample_items_from_candidates(candi_items, sampled_items_num, tgt_item, item_to_remove)
        sampled_items += cur_sampled_items

    return sampled_items


r'''
This function preprocesses the user history file and generates various output files for training, validation, and testing in T4 data format.

Args:
args: An argparse.Namespace object containing the following attributes:
    - infile (str): The input file containing the raw data.
    - item2cate_file (str): The file containing the item2cate information.
    - outdir (str): The directory to store the output files.
    - n_neg_k (int): The number of negative samples for each positive sample (default is 5).

Outputs:
The function generates the following files in the output directory:
    - train.txt
    - valid.txt
    - test.txt
    - user_history.txt
    - user2uid.txt
    - item2tid.txt
    - cate2cid.txt

Example:
    Input data format (user, item1, item2, ..., itemN):
    1 101 102 103
    2 201 202 203 204
    3 301 302 303 304 305

    Input summary:
        n_users  : 3
        n_items  : 12

    Output train samples (user, item_list, label_list):
        1 1,8,12,10,4,6 1,0,0,0,0,0
        2 4,2,9,11,1,10 1,0,0,0,0,0
        2 5,3,1,10,12,2 1,0,0,0,0,0
        ...
    Output valid samples (user, item_list, label_list):
        1 2,4,7,10,11,8 1,0,0,0,0,0
        ...  
    Output test samples (user, item_list, label_list):
        1 3,6,12,11,4,9 1,0,0,0,0,0
        ...
'''
def run_adaranker(arguments):
    wt_train = open(arguments['train_file'], 'w')
    wt_valid = open(arguments['valid_file'], 'w')
    wt_test = open(arguments['test_file'], 'w')
    wt_user_history = open(arguments['user_history_file'], 'w')

    pos_user_history = dict()
    all_item_set = set()
    lengths = []
    train_pos_items = []
    valid_pos_items = []
    test_pos_items = []
    _cnt = 0
    with open(arguments['infile'], 'r') as rd:
        for line in rd.readlines():
            words = line.strip().split(' ')
            user, items = words[0], words[1:]
            item_set = set()
            dedup_items = []
            for item in items:
                if item not in item_set:
                    item_set.add(item)
                    dedup_items.append(item)
            items = dedup_items
            _cnt += len(items)
            item_len = len(items)
            if item_len < 3:
                continue
            lengths.append(item_len)
            train_pos_items.append(items[:-2])  # flatten the train items
            valid_pos_items.append(items[-2])
            test_pos_items.append(items[-1])

            pos_user_history[user] = items
            all_item_set = all_item_set | item_set
    print(f"#. intersections: {_cnt}.")

    sorted_user_list = sorted(set(pos_user_history.keys()), key=int)
    n_users = len(sorted_user_list)
    all_item_list = sorted(all_item_set, key=int)
    n_items = len(all_item_list)

    item2cate = json.load(open(arguments['item2cate_file'], 'r'))
    all_cate_set = set()
    cate2item = {}
    for items in pos_user_history.values():
        for item in items:
            cate = item2cate[item]
            for c in cate:
                if c not in cate2item.keys():
                    cate2item[c] = []
                cate2item[c].append(item)
            all_cate_set = all_cate_set | set(cate)
    cate2item_uni = deepcopy(cate2item)
    for k, v in cate2item_uni.items():
        cate2item_uni[k] = list(set(v))
    all_cate_set = sorted(all_cate_set, key=int)
    n_cates = len(all_cate_set)

    # plus 1 for history zero padding
    user2uid = {user: uid for user, uid in zip(sorted_user_list, range(1, n_users+1))}
    item2tid = {item: tid for item, tid in zip(all_item_list, range(1, n_items+1))}
    cate2cid = {cate: cid for cate, cid in zip(all_cate_set, range(1, n_cates+1))}

    write_dict_to_file(user2uid, arguments['user2uid_file'])
    write_dict_to_file(item2tid, arguments['item2tid_file'])
    write_dict_to_file(cate2cid, arguments['cate2cid_file'])

    def write_to_file(wt, user, items, labels=None):
        uid = user2uid[user]
        tids = [str(item2tid[item]) for item in items]
        data_str = f"{uid} {','.join(tids)}"
        data_str += f" {','.join(labels)}\n" if labels is not None else "\n"
        wt.write(data_str)

    n_neg_k = arguments['n_neg_k']
    labels = ['1'] + ['0'] * n_neg_k
    for (user, hist), itrains, ivalid, itest in tqdm(zip(pos_user_history.items(), train_pos_items, valid_pos_items, test_pos_items), desc="data splitting and negative sampling", total=n_users):
        st_idx = 0 if len(hist) <= 10 else len(hist) - 10
        for j, itrain in enumerate(itrains):
            if j >= st_idx:
                for cate_ in item2cate[itrain]:
                    train_items = [itrain] + distritbuion_mixer_sampling(cate2cid[cate_], itrain, n_cates, cate2item, cate2item_uni, n_neg_k, hist[:j])
                    write_to_file(wt_train, user, train_items, labels)

        for cate_ in item2cate[ivalid]:
            valid_items = [ivalid] + distritbuion_mixer_sampling(cate2cid[cate_], ivalid, n_cates, cate2item, cate2item_uni, n_neg_k, hist[:-2])
            write_to_file(wt_valid, user, valid_items, labels)

        for cate_ in item2cate[itest]:
            test_items = [itest] + distritbuion_mixer_sampling(cate2cid[cate_], itest, n_cates, cate2item, cate2item_uni, n_neg_k, hist[:-1])
            write_to_file(wt_test, user, test_items, labels)

        write_to_file(wt_user_history, user, hist)

    wt_train.close()
    wt_valid.close()
    wt_test.close()
    wt_user_history.close()

    print(f'max item length: {max(lengths)}')
    print(f'min item length: {min(lengths)}')


"""
This function processes a given pandas DataFrame to extract item sequences and the maximum item number.
"""
def load_item_seq(data):
    print(data['item_seq'].apply(max))
    item_num = max(data['item_seq'].apply(max))
    # print('item_num', item_num)
    corpus = []
    for item_seq in tqdm(data.item_seq, desc='load item_seq'):
        seq_ = []
        for token in item_seq:
            seq_.append(str(token))
        corpus.append(seq_)

    return corpus, item_num


"""
This function converts an embedding (a list or array of numbers) into a string.
"""
def emb2str(embedding):
    str_emb = ','.join([str('%.6f' % element) for element in embedding])
    return str_emb


"""
This function pretrains a Word2Vec model using user-item interaction data to get item embeddings.
"""
def pretrain_word2vec(arguments):
    # load data
    print('\nTraining word2vec...')
    infile = arguments['user_history_file']
    outfile_path = arguments['outdir']

    user2itemseq = pd.read_csv(infile, sep=' ', header=None, names=['user_id', 'item_seq'])
    print(user2itemseq)
    user2itemseq.item_seq = user2itemseq.item_seq.apply(lambda t: np.array([int(a) for a in t.split(',')]))
    print(user2itemseq)

    corpus, item_num = load_item_seq(user2itemseq)

    vector_size = arguments['embedding_size']
    w2v_model = Word2Vec(corpus, vector_size=vector_size, window=10, min_count=3, workers=4)
    w2v_model.save(os.path.join(outfile_path, "word2vec.model"))

    fp = open(os.path.join(outfile_path, 'item_emb_'+str(vector_size)+'.txt'), 'w')
    for i in tqdm(range(1, item_num+1), desc='output Item Emb'):
        try:
            item_emb = w2v_model.wv[str(i)]
        except:
            item_emb = np.zeros((vector_size,), dtype=float)
        str_emb = emb2str(item_emb)
        fp.write('{0}{1}{2}\n'.format(i, '\t', str_emb))
    fp.close()


def parse_cmd_arguments():
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--prefile', type=str, default='')
    arguments.add_argument('--infile', type=str)
    arguments.add_argument('--outdir', type=str)
    arguments.add_argument('--n_neg_k', type=int, default=5)
    arguments.add_argument('--data_format', type=str, default='libfm', choices=["libfm", "rank", "adaranker"])
    arguments.add_argument('--sep', type=str, default=",")
    arguments.add_argument('--prefile_file_format', type=str, default="user-item")
    arguments.add_argument('--item2cate_file', type=str)
    arguments.add_argument('--pretrain_word2vec', type=int, default=1)
    arguments.add_argument('--embedding_size', type=int, default=64)

    args = arguments.parse_args()
    # print(args)
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None and value not in ['none', 'None']:
            parsed_results[arg] = value
    return parsed_results


def main(arguments):
    outdir = arguments['outdir']
    os.makedirs(outdir, exist_ok=True)
    arguments['train_file'] = os.path.join(outdir, 'train.txt')
    arguments['valid_file'] = os.path.join(outdir, 'valid.txt')
    arguments['test_file'] = os.path.join(outdir, 'test.txt')
    arguments['user_history_file'] = os.path.join(outdir, 'user_history.txt')
    arguments['user2uid_file'] = os.path.join(outdir, 'user2uid.txt')
    arguments['item2tid_file'] = os.path.join(outdir, 'item2tid.txt')
    arguments['cate2cid_file'] = os.path.join(outdir, 'cate2cid.txt')

    if arguments['prefile']:
        data_formatting(arguments['prefile'], arguments['infile'], arguments['sep'], arguments['prefile_file_format'])

    if arguments['data_format'] == 'libfm':
        arguments['item2hid_file'] = os.path.join(outdir, 'item2hid.txt')
        arguments['rawdata_info_file'] = os.path.join(outdir, 'raw_data.info')

        run_libfm(arguments)

    elif arguments['data_format'] == 'rank':
        run_rank(arguments)

    elif arguments['data_format'] == 'adaranker':
        assert arguments['item2cate_file'] is not None, "You need provide item2cate file for negative sampling in Ada-Ranker"
        run_adaranker(arguments)

    else:
        raise ValueError(f"Unsupported data format: {arguments['data_format']}")

    if arguments['pretrain_word2vec']:
        pretrain_word2vec(arguments)


if __name__ == '__main__':
    arguments = parse_cmd_arguments()
    print(arguments)
    main(arguments)
