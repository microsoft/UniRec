# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.utils.data import Dataset
import numpy as np
from unirec.utils.file_io import * 


class InferDataset(Dataset):
    def __init__(self, config, data_ids, user_history=None, is_seqrec=True):
        self.config = config
        self.data_ids = data_ids
        self.node_type = config.get('node_type', 'user')
        self.last_item = config.get('last_item', 0)
        self.user_history = user_history
        self.empty_history = np.zeros((1,), dtype=np.int32)
        self.is_seqrec = is_seqrec
        self.max_seq_len = config.get('max_seq_len', 0)
        self.use_features = config.get('use_features', 0)
        if self.use_features:
            self.features_num = len(eval(config.get('features_shape', '[]')))
            self.item2features = load_features(self.config['features_filepath'], self.config['n_items'], self.features_num)
        self.set_return_column_index()

    def set_return_column_index(self):
        self.return_key_2_index = {}
        if self.node_type == 'user':
            self.return_key_2_index['user_id'] = len(self.return_key_2_index)
        elif self.node_type == 'item':
            self.return_key_2_index['items'] = len(self.return_key_2_index)
            if self.use_features:
                self.return_key_2_index['item_features'] = len(self.return_key_2_index)
        else:
            raise NotImplementedError('node_type {0} is not supported'.format(self.node_type))
        if self.is_seqrec: 
            self.return_key_2_index['item_seq'] = len(self.return_key_2_index)
            self.return_key_2_index['item_seq_len'] = len(self.return_key_2_index)
            if self.use_features:
                self.return_key_2_index['item_seq_features'] = len(self.return_key_2_index)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        return_tup = (data_id, )
        if self.node_type == 'item' and self.use_features:
            item_features = self.item2features[data_id]
            return_tup = return_tup + (item_features, )
        if self.is_seqrec:
            res = np.zeros((self.max_seq_len,), dtype=np.int32)
            hist = self.user_history[data_id]
            if hist is None:
                hist = self.empty_history
            if self.last_item > 0:
                hist = hist[:-self.last_item]
            n = len(hist)
            if n > self.max_seq_len:
                res[:] = hist[n-self.max_seq_len:]
            else:
                res[self.max_seq_len-n:] = hist[:]
            res_len = min(n, self.max_seq_len)
            return_tup = return_tup + (res, res_len)
            if self.use_features:
                res_features = self.item2features[res]
                return_tup = return_tup + (res_features, )
        return return_tup