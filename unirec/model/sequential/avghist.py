# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import copy    

from .seqrec_base import SeqRecBase

class AvgHist(SeqRecBase):
    
    def __init__(self, config):
        self.asymmetric = config['asymmetric']
        self.alpha = config['user_sequence_alpha']
        super(AvgHist, self).__init__(config)

    def _define_model_layers(self):
        if self.asymmetric:
            self.item_src_embedding = self.item_embedding
            self.item_dst_embedding = copy.deepcopy(self.item_embedding)
        else:
            self.item_src_embedding = self.item_embedding 
            self.item_dst_embedding = self.item_embedding 
    
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
        item_seq_emb = self.item_embedding_for_user(item_seq, item_seq_features, time_seq) # [2048, 100, 64]
        if self.time_seq:
            time_embedding = self.time_embedding(time_seq)
            item_seq_emb = item_seq_emb + time_embedding
        coeff = torch.pow((item_seq_len+1).float(), -self.alpha).unsqueeze(1)
        user_emb = coeff * item_seq_emb.sum(1)   

        return user_emb 

    def item_embedding_for_user(self, item_seq, item_seq_features=None, time_seq=None):
        item_emb = self.item_dst_embedding(item_seq)
        if self.use_features:
            item_features_emb = self.features_embedding(item_seq_features).sum(-2)
            item_emb = item_emb + item_features_emb
        if self.time_seq:
            time_embedding = self.time_embedding(time_seq)
            item_emb = item_emb + time_embedding
        if self.use_text_emb:
            text_emb = self.text_mlp(self.text_embedding(item_seq))
            item_emb = item_emb + text_emb
        return item_emb
 