# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from ..base.ranker import Ranker
import unirec.model.modules as modules

class BST(Ranker):    
    def __init__(self, config):
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = float(config['layer_norm_eps'])
        self.max_seq_len = config['max_seq_len']
        self.seq_decay = config['seq_decay']
        super(BST, self).__init__(config)

    def _define_model_layers(self): 
        # multi-head attention
        self.position_embedding = nn.Embedding(self.max_seq_len+1, self.hidden_size)
        self.trm_encoder = modules.TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )   

    def _get_attention_mask(self, item_seq):
        """Generate bi-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long() #(batch_size, seq_len)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # torch.int64, (batch_size, 1, seq_len, 1)
        subsequent_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def forward_scores(self, item_id=None, item_features=None, item_seq=None, item_seq_len=None,  item_seq_features=None):
        # item_id: (batch_size, n_inline_items), item_features: (batch_size, n_inline_items, n_features), 
        # item_seq: (batch_size, seq_len), item_seq_len: (batch_size),  item_seq_features: (batch_size, seq_len, n_features)
        n_inline_items = -1
        if item_id.dim() == 2:
            n_inline_items = item_id.shape[1]
            item_id = item_id.reshape(-1)
            item_seq = item_seq.unsqueeze(1).repeat(1,n_inline_items,1).reshape(-1, item_seq.shape[-1])
            item_seq_len = item_seq_len.unsqueeze(1).repeat(1,n_inline_items).reshape(-1)
            if self.use_features:
                item_features = item_features.reshape(-1, item_features.shape[-1])
                item_seq_features = item_seq_features.unsqueeze(1).repeat(1, n_inline_items,1,1).reshape(-1, *item_seq_features.shape[-2:])
        item_emb = self.item_embedding(item_id)
        seq_emb = self.item_embedding(item_seq)
        if self.use_features:
            item_features_emb = self.features_embedding(item_features).sum(-2)
            item_seq_features_emb = self.features_embedding(item_seq_features).sum(-2)
            item_emb = item_emb + item_features_emb
            seq_emb = seq_emb + item_seq_features_emb
        if self.use_text_emb:
            item_text_emb = self.text_mlp(self.text_embedding(item_id))
            seq_text_emb = self.text_mlp(self.text_embedding(item_seq))
            item_emb = item_emb + item_text_emb
            seq_emb = seq_emb + seq_text_emb
        input_emb = torch.cat([seq_emb, item_emb.unsqueeze(1)], dim=1)
        new_seq = torch.cat([item_seq, item_id.unsqueeze(1)], dim=1)

        position_ids = torch.arange(new_seq.size(1), dtype=torch.long, device=item_id.device)
        position_ids = position_ids.unsqueeze(0).expand_as(new_seq)
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_emb + position_embedding
        
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self._get_attention_mask(new_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        non_zero_num = item_seq_len.unsqueeze(-1) + 1
        output = (output * torch.logspace(self.seq_decay, 0, steps=self.max_seq_len+1).unsqueeze(0).unsqueeze(-1).to(output.device)).sum(1) / non_zero_num.pow(0.5)
        
        output = self.output_layer(output).squeeze(-1)
        if self.has_item_bias:
            output = output + self.item_bias[item_id]
        if n_inline_items > 0:
            output = output.reshape(-1, n_inline_items)
        return output  # [B]


     