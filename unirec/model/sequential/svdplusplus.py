# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import copy  
import torch.nn as nn  

from .seqrec_base import SeqRecBase
from unirec.utils import general

class SVDPlusPlus(SeqRecBase):
    
    def __init__(self, config):
        self.alpha = config['user_sequence_alpha']
        super(SVDPlusPlus, self).__init__(config)

    def _define_model_layers(self):
        self.item_src_embedding = self.item_embedding
        self.item_dst_embedding = copy.deepcopy(self.item_embedding) 
    
    def forward_item_emb(self, items, item_features=None):       
        item_emb = self.item_src_embedding(items) # [batch_size, n_items_inline, embedding_size]
        if self.use_features:
            item_features_emb = self.features_embedding(item_features).sum(-2)
            item_emb = item_emb + item_features_emb
        if self.use_text_emb:
            text_emb = self.text_mlp(self.text_embedding(items))
            item_emb = item_emb + text_emb
        return item_emb 

    def forward_user_emb(self, user_id=None, item_seq=None, item_seq_len=None, item_seq_features=None, time_seq=None):
        self.user = user_id
        user_emb01 = self.user_embedding(self.user)
        
        user_emb02 = self.item_dst_embedding(item_seq)  # batch_size x MAX_SEQ_LEN x embedding_size
        coeff = torch.pow((item_seq_len+1).float(), -self.alpha).unsqueeze(1) # batch_size x 1 
        user_emb = user_emb01 + coeff * (user_emb02.sum(1))   

        return user_emb 
