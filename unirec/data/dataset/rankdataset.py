# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging 
import time
from datetime import datetime 
import argparse
import numpy as np
import pandas as pd
import pickle as pkl 
import torch
from torch.utils.data import DataLoader, Dataset
from unirec.constants.protocols import DataFileFormat

from unirec.utils.file_io import *
from .basedataset import BaseDataset


class RankDataset(BaseDataset):
    def __init__(self, config, path, filename, transform=None):
        super(RankDataset, self).__init__(config, path, filename, transform) 
        self.add_seq_transform = None
        # if the group_size > 0, the raw data will be collected by group, then by batch.
        self.group_size = self.config['group_size'] if self.config['group_size'] > 0 else 1

    def __len__(self):
        _len = super(RankDataset, self).__len__()
        if _len % self.group_size != 0:
            assert ValueError(f"The data can not be divided into groups with data_size: {_len}, group_size: {self.group_size}!")
        return _len // self.group_size

    def __getitem__(self, index):
        elements = []
        start_index = self.group_size * index
        for _index in range(start_index, start_index + self.group_size):
            _elements = super(RankDataset, self).__getitem__(_index)  # index_list, value_list, label, ...
            elements.append(_elements)
        
        elements_left = tuple(np.array(t) for i, t in enumerate(zip(*elements)) if i >= 2)

        index_list = [t[0] for t in elements]
        value_list = [t[1] for t in elements]

        elements = (index_list, value_list)
        if self.group_size == 1:
            elements += (elements_left[0],)
            if elements_left[1:]:
                elements += (t.squeeze() for t in elements_left[1:])
        else:
            elements += elements_left
        return elements