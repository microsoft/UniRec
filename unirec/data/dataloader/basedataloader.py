import numpy as np
import torch

class BaseDataLoader(object):  
    def __init__(self, dataset, batch_size=256, **kwargs):
        self.ptr = 0
        self.batch_size = batch_size
        self.N_raw_dataset = len(dataset)
        self.N = (self.N_raw_dataset - 1) // batch_size + 1
        self.dataset = dataset
        
        self._set_return_buffer()
    
    def _set_return_buffer(self):
        record = self.dataset.__getitem__(0)
        m = len(record)
        
        data = []
        for i in range(m): 
            if isinstance(record[i], int):
                data.append(np.zeros((self.batch_size,), dtype=np.int64))
            elif isinstance(record[i], float):
                data.append(np.zeros((self.batch_size,), dtype=np.float32))
            else:
                data.append(np.zeros((self.batch_size, len(record[i])), dtype=record[i].dtype))
        self.return_buffer = data
    
    def __len__(self):
        return self.N
    
    def __iter__(self):
        self.ptr = 0
        return self
    
    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        start = self.ptr * self.batch_size
        end = min(self.N_raw_dataset, start + self.batch_size) 
        self.ptr += 1
        res = []
        m = 0
        for i in range(start, end):
            record = self.dataset.__getitem__(i)
            m = len(record)
            for j in range(m):
                self.return_buffer[j][i-start]=record[j]
        for j in range(m):
            res.append(torch.from_numpy(self.return_buffer[j][:(end-start)]))  
        return res