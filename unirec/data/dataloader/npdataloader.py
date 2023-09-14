import numpy as np

class NPDataLoader(object):  
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
                data.append(np.zeros((self.batch_size,), dtype=object))
        self.return_buffer = data
        self.M_cols = m
    
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
        for i in range(start, end):
            record = self.dataset.__getitem__(i) 
            for j in range(self.M_cols):
                self.return_buffer[j][i-start]=record[j]
        for j in range(self.M_cols):
            res.append(self.return_buffer[j][:(end-start)])
        return res