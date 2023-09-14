import numpy as np
import torch
import pandas as pd
import copy
import random
from unirec.utils import general
from unirec.constants.protocols import *

class AddUserHistory(object):
    r'''
    Parameters:
        user2history: An N-length ndarray (N is user count) of ndarray (of history items). 
        mask_mode: If the item history is a sequence, we suggest to use autoregressive to avoid use future inforamtion.
                    Otherwise use unorder.
    '''
    def __init__(self, user2history, mask_mode='unorder', user2history_time=None, seq_last=0): 
        self.user2history = user2history
        self.user2history_time = user2history_time
        self.empty_history = np.zeros((1,), dtype=np.int32)
        self.mask_mode = mask_mode
        self.seq_last = seq_last

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
        items = sample[1] 
        if isinstance(items, list) or isinstance(items, np.ndarray):
            items = set(items)
        else:
            items = set([items])
        userid = sample[0]
        if userid >= len(self.user2history) or self.user2history[userid] is None:
            history = self.empty_history
            history_time = self.empty_history if self.user2history_time is not None else None
        else:
            history = self.user2history[userid]
            history_time = self.user2history_time[userid] if self.user2history_time is not None else None
        
        if self.mask_mode == HistoryMaskMode.Unorder.value:
            history = copy.deepcopy(history)
            for idx, item in enumerate(history):
                if item in items:
                    history[idx] = 0
                    if self.user2history_time is not None:
                        history_time[idx] = 0
        elif self.mask_mode == HistoryMaskMode.Autoregressive.value:
            n = []
            for idx, item in enumerate(history):
                if item in items:
                    n.append(idx)
            if len(n) == 0:
                pass
            else:
                n = n[-1] if self.seq_last else random.choice(n)
                history = history[:n]
                history_time = history_time[:n] if self.user2history_time is not None else None
        
        return history, len(history), history_time
        