import numpy as np
import torch
import pandas as pd
import random
from copy import deepcopy

from unirec.utils.sampling import prepare_aliased_randomizer

class AddNegSamples(object):
    r"""
       Optional parameters:
            user2history: A N-len ndarray to store users' interacted items.
            item_popularity: A M-len ndarray to store items' popularity. 
    """
    def __init__(self, n_users, n_items, n_neg, **kwargs):
        self.n_users = n_users
        self.n_items = n_items
        self.n_neg = n_neg

        self.user2history_as_set = None
        self.item2sample_ratio = None
        self.neg_by_pop_alpha = 1.0

        for k,v in kwargs.items():
            if k == 'user2history' and v is not None:
                self.user2history_as_set = self._construct_history(v)
            elif k == 'item_popularity' and v is not None:
                self.item2sample_ratio = v ## placeholder, will construct the true sampling weights latter
            elif k == 'neg_by_pop_alpha' and v is not None:
                self.neg_by_pop_alpha = v
        
        if self.item2sample_ratio is not None:
            self.item2sample_ratio = self._construct_item_sample_ratio(self.item2sample_ratio, self.neg_by_pop_alpha)
            self.item_sampler = prepare_aliased_randomizer(self.item2sample_ratio)
    
    r"""
        Convert ndarray to set for fast item checking.
        Parameters:
            user2history: A ndarray of ndarray.
        Returns:
            user2history_as_set: A ndarray of set.
    """
    def _construct_history(self, user2history):
        N = len(user2history)
        user2history_as_set = np.empty(N, dtype=object)
        for i in range(N):
            if user2history[i] is not None and len(user2history[i]) > 0:
                if isinstance(user2history[i], set):
                    user2history_as_set[i] = user2history[i]
                else:
                    user2history_as_set[i] = set(user2history[i])
        return user2history_as_set

    r"""
        Parameters:
            item2popularity: an M-len ndarray, reflects the number of positive samples for the item in training set. 
    """
    def _construct_item_sample_ratio(self, item2popularity, alpha=0.5):
        res = np.power(item2popularity, alpha)
        res /= np.sum(res)
        res[0] = 0  ## item_id 0 is a fake item, should not be sampled
        return res

    r"""
        Check the rationality of selected_item_id as a negative item.
    """
    def _valid(self, user_id, pos_item_id, selected_item_id):
        pos_set =  set(pos_item_id) if isinstance(pos_item_id, np.ndarray) else set([pos_item_id])
        if selected_item_id in pos_set:
            return False 
        if self.user2history_as_set is None or self.user2history_as_set[user_id] is None or selected_item_id not in self.user2history_as_set[user_id]:
            return True
        return False

    def _sample_one_item(self):
        if self.item2sample_ratio is None:
            return random.randint(1, self.n_items - 1)

        # return np.random.choice(self.n_items, size=1, replace=True, p=self.item2sample_ratio)[0]
        return self.item_sampler() 

    def get_fake_label(self, k):
        if hasattr(self, 'fake_label'):
            return self.fake_label  
        res = np.zeros((k,), dtype=np.int32)
        res[0] = 1  
        self.fake_label = res
        return res 

    def __call__(self, sample): 
        ## suppose only one positive item;  
        ## sample is a ndarray of object : [userid, itemid, label, ...] 
        sample = deepcopy(sample)
        pos_item = sample[1] 
        pos_len = len(pos_item) if isinstance(pos_item, np.ndarray) else 1
        
        sampled_items = np.zeros(self.n_neg+pos_len, dtype=int) 
        for i in range(pos_len, self.n_neg+pos_len):
            retries = 100
            sampled_itemid = 0
            while retries > 0:
                idx = self._sample_one_item()
                if self._valid(sample[0], pos_item, idx):
                    sampled_itemid = idx
                    break
                retries -= 1
            sampled_items[i] = sampled_itemid

        sampled_items[0:pos_len] = pos_item 
        sample[1] = sampled_items
        if len(sample) >= 3: # I think if there exists label, it should not be replaced by fake label
            all_labels = np.zeros((self.n_neg+pos_len,), dtype=np.int32)
            all_labels[0:pos_len] = sample[2]
            sample[2] = all_labels
        return sample
        