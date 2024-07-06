# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import inspect
from .reco_abc import AbstractRecommender


class Ranker(AbstractRecommender): 
    def add_annotation(self):
        super(Ranker, self).add_annotation()
        self.annotations.append('Ranker')

    def _define_model_layers(self):
        pass

    def forward_scores(self, item_id=None, item_features=None, item_seq=None, item_seq_len=None, item_seq_features=None):
        raise NotImplementedError

    def forward(self, user_id=None, item_id=None, label=None, item_features=None, item_seq=None, item_seq_len=None, item_seq_features=None, time_seq=None, session_id=None, reduction=True, return_loss_only=True, max_len=None):
        scores = self.forward_scores(item_id, item_features, item_seq, item_seq_len,  item_seq_features)
        if self.SCORE_CLIP > 0:
            scores = torch.clamp(scores, min=-1.0*self.SCORE_CLIP, max=self.SCORE_CLIP) 
        if self.training:
            loss = self._cal_loss(scores, label, reduction)  
            if return_loss_only:
                return loss, None, None, None
            return loss, scores, None, None # because we don't have user embedding and item embedding
        else:
            return None, scores, None, None

    def predict(self, interaction):
        inputs = {k: v for k, v in interaction.items() if k in inspect.signature(self.forward_scores).parameters}
        scores = self.forward_scores(**inputs)
        if self.SCORE_CLIP > 0:
            scores = torch.clamp(scores, min=-1.0*self.SCORE_CLIP, max=self.SCORE_CLIP)
        return scores.detach().cpu().numpy()
