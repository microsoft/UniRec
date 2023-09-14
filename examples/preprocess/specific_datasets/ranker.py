
import argparse
import json
import os
import random

import pandas as pd


r'''
This function preprocesses the single column user history file and generates sequential user history file.
'''
def data_formatting(prefile, outfile, sep: str=",", input_file_format: str="user-item"):
    assert input_file_format in {"user-item", "user-item_seq"}, "`input_file_format` must be `user-item` or `user-item_seq`"
    if input_file_format == "user-item":
        df = pd.read_csv(prefile, header=0, names=['user_id', 'item_id'], dtype={'user_id': int, 'item_id': str}, sep=sep)

        df['item_id'] = df['item_id'].apply(lambda x: [x])
        df_grouped = df.groupby('user_id').agg({'item_id': lambda x: [item[0] for item in x.drop_duplicates()]})
    else:
        df = pd.read_csv(prefile, header=0, names=['user_id', 'item_seq'], dtype={'user_id': int, 'item_seq': str}, sep=sep)
        df['item_id'] = df['item_seq'].apply(lambda x: x.split(","))
        df_grouped = df[['user_id', 'item_id']]
    user_items_dict = df_grouped.to_dict()['item_id']
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

        write_to_file(wt_user_history, user, hist)

    wt_train.close()
    wt_valid.close()
    wt_test.close()
    wt_user_history.close()

    print(f'max item length: {max(lengths)}')
    print(f'min item length: {min(lengths)}')


def parse_cmd_arguments():
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--prefile', type=str, default='')
    arguments.add_argument('--infile', type=str)
    arguments.add_argument('--outdir', type=str)
    arguments.add_argument('--n_neg_k', type=int, default=5)
    arguments.add_argument('--data_format', type=str, default='libfm', choices=["libfm", "rank"])
    arguments.add_argument('--sep', type=str, default=",")
    arguments.add_argument('--prefile_file_format', type=str, default="user-item")

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

    if arguments['prefile']:
        data_formatting(arguments['prefile'], arguments['infile'], arguments['sep'], arguments['prefile_file_format'])

    if arguments['data_format'] == 'libfm':
        arguments['item2hid_file'] = os.path.join(outdir, 'item2hid.txt')
        arguments['rawdata_info_file'] = os.path.join(outdir, 'raw_data.info')

        run_libfm(arguments)

    elif arguments['data_format'] == 'rank':
        run_rank(arguments)

    else:
        raise ValueError(f"Unsupported data format: {arguments['data_format']}")


if __name__ == '__main__':
    arguments = parse_cmd_arguments()
    print(arguments)
    main(arguments)
