# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from ..base.recommender import BaseRecommender
from unirec.constants.protocols import ColNames

class MultiVAE(BaseRecommender):
    r""" MultiVAE extends variational autoencoders (vaes) to collaborative filtering
        for implicit feedback. For more details, please refer to https://arxiv.org/pdf/1802.05814.pdf
    """
    def __init__(self, config):
        super(MultiVAE, self).__init__(config)

    
    def add_annotation(self):
        super(MultiVAE, self).add_annotation()
        self.annotations.append('AERecBase')


    def _define_model_layers(self):
        self.anneal_cap = self.config['anneal_cap']
        self.total_annel_steps = self.config['total_anneal_steps']
        self.anneal = 0.0
        self.encoder_dims = self.config['encoder_dims']
        self.decoder_dims = self.config['decoder_dims']
        #
        # encoder layer, actually a MLP layer
        _encoder_mlp_dims = [self.embedding_size, ] + self.encoder_dims[:-1] + [self.encoder_dims[-1] * 2]
        self.encoder_layers = nn.Sequential()
        for i, (d_in, d_out) in enumerate(zip(_encoder_mlp_dims[:-1], _encoder_mlp_dims[1:])):
            self.encoder_layers.append(nn.Linear(d_in, d_out))
            if i != len(_encoder_mlp_dims)-2:
                self.encoder_layers.append(nn.Tanh())
        #
        # decoder layer
        _decoder_mlp_dims = [self.encoder_dims[-1], ] + self.decoder_dims + [self.embedding_size]
        self.decoder_layers = nn.Sequential()
        for i, (d_in, d_out) in enumerate(zip(_decoder_mlp_dims[:-1], _decoder_mlp_dims[1:])):
            self.decoder_layers.append(nn.Linear(d_in, d_out))
            if i != len(_decoder_mlp_dims)-1:
                self.decoder_layers.append(nn.Tanh())
        #
        self.dropout = nn.Dropout(self.dropout_prob)


    def forward_user_emb(self, user_id=None, item_seq=None, item_seq_len=None, item_seq_features=None, time_seq=None):
        inter_hist_emb = self.item_embedding_for_user(item_seq, item_seq_features, time_seq)
        nnz = item_seq.count_nonzero(dim=1).unsqueeze(-1)
        inter_hist_emb = inter_hist_emb.sum(1) / (nnz.pow(0.5) + torch.finfo(torch.float32).eps)
        en_h = self.encoder_layers(torch.tanh(self.dropout(inter_hist_emb)))
        #
        mu, logvar = en_h.tensor_split(2, dim=-1)
        z = self._reparameterize(mu, logvar)
        de_h = self.decoder_layers(z)
        if self.training:
            return de_h, mu, logvar
        else:
            return de_h

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            # In evaluation, to provide diversity(novelty), reparameterize tricks could be used as well.
            # Here `sampling_times` controls the times to sample from the normalize distrobution.
            sampling_times = self.config['eval_reparameter_sampling_times']
            if sampling_times > 0:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn((*std.shape, sampling_times), device=std.device)  # [B, D, n]
                eps = eps.mean(-1)
                output = eps.mul(std).add_(mu)
                return output
            else:
                return mu

    def _predict_layer(self, user_emb, items_emb, user_id=None, item_id=None):
        if user_emb.shape != items_emb.shape:
            if user_emb.dim() == items_emb.dim():
                scores = user_emb @ items_emb.T
            else:
                user_emb = torch.repeat_interleave(
                    user_emb, items_emb.shape[-2], dim=-2
                )
                user_emb = user_emb.reshape(items_emb.shape) 
                scores = torch.mul(user_emb, items_emb).sum(dim=-1)
        else:
            scores = torch.mul(user_emb, items_emb).sum(dim=-1)
        
        return scores

    def forward(self, user_id=None, item_id=None, label=None, item_features=None, item_seq=None, item_seq_len=None, item_seq_features=None, time_seq=None, session_id=None, reduction=True, return_loss_only=True):
        # items_emb = self.item_embedding.weight
        in_item_id = torch.arange(self.n_items).to(self.device)
        in_item_features = torch.tensor(self.item2features, dtype=torch.int32).to(self.device) if self.use_features else None
        items_emb = self.forward_item_emb(in_item_id, in_item_features)
        
        user_emb, mu, logvar = self.forward_user_emb(user_id, item_seq, item_seq_len, item_seq_features, time_seq)
        all_scores = self._predict_layer(user_emb, items_emb)
        label = item_seq
        softmax_loss = self.softmax_loss(all_scores, label)
        kl_loss = - 0.5 * torch.mean(torch.sum(1+logvar-mu.pow(2) - logvar.exp(), dim=1))
        kl_loss = self.anneal * kl_loss
        loss = softmax_loss + kl_loss
        # update anneal
        self.anneal = min(self.anneal_cap, self.anneal + (1.0 / self.total_annel_steps))  
        if return_loss_only:
            return loss, None, None, None
        return loss, all_scores, user_emb, items_emb    

    def softmax_loss(self, all_scores, labels):
        pos_scores = torch.gather(all_scores, -1, labels.long())
        softmax_loss = - pos_scores + torch.logsumexp(all_scores, dim=-1, keepdim=True)
        softmax_loss[labels==0] = 0.0
        nnz = (labels != 0).sum()
        softmax_loss = softmax_loss.sum() / nnz
        return softmax_loss

