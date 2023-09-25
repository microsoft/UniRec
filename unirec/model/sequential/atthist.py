# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn

from .seqrec_base import SeqRecBase
import unirec.model.modules as modules

class AttHist(SeqRecBase):    
    def __init__(self, config):
        super(AttHist, self).__init__(config)

    def _define_model_layers(self):
        self.attention = modules.AttentionMergeLayer(self.embedding_size, self.dropout_prob)
        self.emb_dropout = nn.Dropout(self.dropout_prob) 

    def forward_user_emb(self, user_id=None, item_seq=None, item_seq_len=None, item_seq_features=None, time_seq=None):
        item_seq_emb = self.item_embedding_for_user(item_seq, item_seq_features, time_seq)    
        
        ## actually, there is no sequence informace in this model
        seq_emb = self.attention(item_seq_emb)  
        return seq_emb
    
