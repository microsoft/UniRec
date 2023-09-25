# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect

import torch
import torch.nn as nn
from torch.nn.init import constant_, normal_

import unirec.model.modules as modules

from ..base.ranker import Ranker


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features=1, linear_mode='gather'):
        super(SparseLinear, self).__init__()
        assert linear_mode in ['gather', 'full', 'sparse', 'embedding'], f"SparseLinear mode - {linear_mode} is not supported."
        self.linear_mode = linear_mode
        self.in_features = in_features

        if self.linear_mode != 'embedding':
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            constant_(self.weight, 0)
            # normal_(self.weight, mean=0, std=0.01)
        else:
            self.fc = nn.Embedding(in_features, out_features)
            constant_(self.fc.weight, 0)

        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, index_list, value_list):
        batch_size, n_inline_feats = index_list.shape[:2]
        if self.linear_mode != 'embedding':
            index_list = index_list.to(torch.long)

        if self.linear_mode == 'gather':  # fastest
            # expand zero-dim of self.weight
            # [1, n_feats] -> [batch_size, n_feats]
            weight_expanded = self.weight.expand(batch_size, -1)
            # gather the weight_factor w.r.t. index_list
            # [batch_size, n_inline_feats]
            weight_gathered = torch.gather(weight_expanded, 1, index_list.to(torch.long))

            # weight_gathered = self.weight[0, index_list]
            products = weight_gathered * value_list
            row_sum = torch.sum(products, dim=1, keepdim=True)
            output = row_sum + self.bias

        elif self.linear_mode == 'full':  # slow
            # generate a full matrix with sparse values
            sparse_tensor = torch.zeros((batch_size, self.in_features), dtype=value_list.dtype).to(value_list.device)
            for i, (index, value) in enumerate(zip(index_list, value_list)):
                sparse_tensor[i, index] = value
            output = sparse_tensor @ self.weight.t() + self.bias

        elif self.linear_mode == 'sparse':  # fast
            # transform index_list to sparse tensor indices
            row_index = torch.arange(batch_size).to(index_list.device)
            _index_list = torch.stack([torch.repeat_interleave(row_index, n_inline_feats), index_list.flatten()], dim=0)
            sparse_tensor = torch.sparse_coo_tensor(_index_list, value_list.flatten(), size=(batch_size, self.in_features))
            output = torch.sparse.mm(sparse_tensor, self.weight.t()) + self.bias

        elif self.linear_mode == 'embedding':  # moderately fast
            linear_emb = self.fc(index_list).squeeze(-1)
            products = linear_emb * value_list
            row_sum = torch.sum(products, dim=1, keepdim=True)
            output = row_sum + self.bias

        return output


class FM(Ranker):
    def __init__(self, config):
        self.n_feats = config['n_feats']
        self.linear_mode = config['linear_mode']
        super(FM, self).__init__(config)

    def _define_model_layers(self):
        del self.item_embedding

        self.fm_linear = SparseLinear(self.n_feats, 1, linear_mode=self.linear_mode)
        self.fm_embedding = nn.Embedding(self.n_feats, self.embedding_size, padding_idx=0)

        normal_(self.fm_embedding.weight, mean=0, std=0.01)

    def forward_scores(self, index_list=None, value_list=None):
        # index_list: (batch_size, n_inline_feats) or (batch_size, group_size, n_inline_feats)
        # value_list: (batch_size, n_inline_feats) or (batch_size, group_size, n_inline_feats)
        n_inline_feats = index_list.shape[-1]
        if index_list.dim() == 3 and index_list.shape[1] == self.group_size:
            index_list = index_list.reshape([-1, n_inline_feats])
            value_list = value_list.reshape([-1, n_inline_feats])

        # first order: [batch_size, 1]
        linear_output = self.fm_linear(index_list, value_list)

        # second order
        # [batch_size, n_inline_feats, embedding_size]
        feat_emb = self.fm_embedding(index_list)
        product = feat_emb * value_list.unsqueeze(-1)

        # [batch_size, embedding_size]
        sum_of_square = torch.sum(product ** 2, dim=1)
        square_of_sum = torch.sum(product, dim=1) ** 2
        # [batch_size, 1]
        fm_output = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        # FM output
        output = linear_output + fm_output

        if self.group_size > 0:
            output = output.reshape([-1, self.group_size])
        return output  # [B, 1] or [B, group_size]

    def forward(self, index_list=None, value_list=None, label=None, reduction=True, return_loss_only=True):
        scores = self.forward_scores(index_list, value_list)
        if self.SCORE_CLIP > 0:
            scores = torch.clamp(scores, min=-1.0*self.SCORE_CLIP, max=self.SCORE_CLIP) 
        if self.training:
            loss = self._cal_loss(scores, label, reduction)  
            if return_loss_only:
                return loss, None, None, None
            return loss, scores, None, None  # because we don't have user embedding and item embedding
        else:
            return None, scores, None, None

    def predict(self, interaction):
        scores = super(FM, self).predict(interaction)
        scores = modules.sigmoid_np(scores)
        return scores

    def load(self, model_file):
        """Imports parameters from an xlearn-generated Factorization Machine (FM) model file to set weights and biases for a Torch-FM model. The xlearn FM model file ends with `.txt` and its format consists of:
            Bias: `bias: value`.
            Linear weights: `i_n: value`, where n equals to `self.n_feats-1`.
            Embedding weights: `v_n: value1 value2 ... valueK`, where K equals to `self.embedding_size`.
        """
        with open(model_file) as rd:
            lines = rd.readlines()

        bias = float(lines[0].strip().split(": ")[1])
        weight = [float(line.strip().split(": ")[1]) for line in lines[1:self.n_feats+1]]
        embedding_weight = [[float(value) for value in line.strip().split(": ")[1].split()] for line in lines[self.n_feats+1:]]

        with torch.no_grad():
            self.fm_linear.bias.data = torch.tensor(bias)
            if self.linear_mode != 'embedding':
                self.fm_linear.weight.data = torch.tensor(weight).view(1, -1)
            else:
                self.fm_linear.fc.weight.data = torch.tensor(weight).view(-1, 1)
            self.fm_embedding.weight.data = torch.tensor(embedding_weight)