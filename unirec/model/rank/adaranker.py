import torch
import torch.nn as nn

from ..base.ranker import Ranker
import unirec.model.modules as modules


r"""
AdaRanker is a model for sequential recommendation, which is implemented based on PyTorch. It's designed to adapt to changes in data distribution.
AdaRanker can choose either GRU or SASRec as the base model, and it modulates the input and encodes it using the chosen base model.

AdaRanker implements a data distribution adaptive ranking model, which can embed users and candidate items, and predict the scores of users for items through forward propagation. The extraction of the distribution vector and the modulation of the input are the core parts of AdaRanker.

For a more detailed description of the AdaRanker model, you can refer to the paper: "Ada-Ranker: A Data Distribution Adaptive Ranking Paradigm for Sequential Recommendation" available at https://arxiv.org/pdf/2205.10775.pdf
"""
class AdaRanker(Ranker):
    def __init__(self, config):
        self.train_type = config['train_type']
        self.base_model = config['base_model']
        assert self.base_model in ["GRU", "SASRec"], "we only support 'GRU' and 'SASRec' as base model in AdaRanker now."

        if self.base_model == "SASRec":
            self.n_layers = config['n_layers']
            self.n_heads = config['n_heads']
            self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
            self.hidden_dropout_prob = config['hidden_dropout_prob']
            self.attn_dropout_prob = config['attn_dropout_prob']
            self.hidden_act = config['hidden_act']
            self.layer_norm_eps = config['layer_norm_eps']
            self.max_seq_len = config['max_seq_len']
            self.use_pos_emb = config['use_position_emb']

        super(AdaRanker, self).__init__(config)

        self.dnn_input_size = self.embedding_size * 2
        self.dnn_inner_size = self.embedding_size

        if self.train_type == 'Ada-Ranker':
            # extract distribution layers
            self.extract_distribution_layer = modules.NeuProcessEncoder(
                self.embedding_size, self.embedding_size, self.embedding_size, 
                self.dropout_prob, self.device
            )

            # add bias layers
            self.film_affine_emb_scale = nn.Linear(self.embedding_size, 1)
            self.film_affine_emb_bias = nn.Linear(self.embedding_size, 1)

            # predict_layer
            self.mem_w1 = modules.MemoryUnit(self.dnn_input_size, self.dnn_inner_size, self.embedding_size)
            self.mem_b1 = modules.MemoryUnit(1, self.dnn_inner_size, self.embedding_size)
            self.mem_w2 = modules.MemoryUnit(self.dnn_inner_size, 1, self.embedding_size)
            self.mem_b2 = modules.MemoryUnit(1, 1, self.embedding_size)
            seq = [
                nn.Dropout(p=self.dropout_prob), 
                modules.AdaLinear(self.dnn_input_size, self.dnn_inner_size),
                nn.Tanh(),
                modules.AdaLinear(self.dnn_inner_size, 1),
            ]
        else:
            seq = [
                nn.Dropout(p=self.dropout_prob), 
                nn.Linear(self.dnn_input_size, self.dnn_inner_size),
                nn.Tanh(),
                nn.Linear(self.dnn_inner_size, 1),
            ]

        self.mlp_layers = torch.nn.Sequential(*seq).to(self.device)

    def _define_model_layers(self):
        if self.base_model == "GRU":
            # gru4rec
            self.hidden_size = self.embedding_size * 2
            self.num_layers = 1
            self.emb_dropout = nn.Dropout(self.dropout_prob)
            self.gru_layers = nn.GRU(
                input_size=self.embedding_size, hidden_size=self.hidden_size,
                num_layers=self.num_layers, bias=True, batch_first=True,
            )
            self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        elif self.base_model == "SASRec":
            # multi-head attention
            self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_size) if self.use_pos_emb else None
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

        else:
            raise AttributeError(f"We do not support base model: {self.base_model} in AdaRanker.")

    def _get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        if self.use_pos_emb:
            # mask for left-to-right unidirectional
            max_len = attention_mask.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long().to(item_seq.device)

            extended_attention_mask = extended_attention_mask * subsequent_mask

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward_user_emb(self, item_seq=None):
        ## TODO: use item_seq, item_seq_features and time_seq to generate item_embedding_for_user
        item_seq_emb = self.item_embedding(item_seq)
        if self.train_type == "Ada-Ranker":
            item_seq_emb = self._input_modulation(item_seq_emb, self.distribution_vector)

        if self.base_model == "GRU":
            item_seq_emb = self.emb_dropout(item_seq_emb)

            gru_output, _ = self.gru_layers(item_seq_emb)
            gru_output = self.dense(gru_output)

            output = gru_output[:, -1]  # [batch_size, embedding_size]
        
        elif self.base_model == "SASRec":
            input_emb = item_seq_emb
            if self.position_embedding is not None:
                position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
                position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
                position_embedding = self.position_embedding(position_ids)
                input_emb = input_emb + position_embedding

            input_emb = self.LayerNorm(input_emb)
            input_emb = self.dropout(input_emb)

            extended_attention_mask = self._get_attention_mask(item_seq)

            trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
            output = trm_output[-1]
            output = output[:, -1, :]  # [B H]

        return output

    def forward_scores(self, item_id=None, item_features=None, item_seq=None, item_seq_len=None, item_seq_features=None):
        # print(item_id.shape, item_seq.shape)
        ## TODO: use item_features to generate item_embedding
        candidate_items_emb = self._get_candidates_emb(item_id)

        if self.train_type == 'Ada-Ranker':
            self.distribution_vector = self._distribution_vector_etractor(candidate_items_emb)  # [batch_size, embedding_size]

        user_emb = self.forward_user_emb(item_seq)
        logits = self._predict_layer(user_emb, candidate_items_emb)

        return logits

    def _predict_layer(self, seq_emb, candidate_items_emb):
        if seq_emb.shape != candidate_items_emb.shape:
            seq_emb = torch.repeat_interleave(
                seq_emb, candidate_items_emb.shape[-2], dim=-2
            )
            seq_emb = seq_emb.reshape(candidate_items_emb.shape)

        seq_emb = torch.cat([seq_emb, candidate_items_emb], -1)

        if self.train_type == 'Ada-Ranker':  # parameter modulation
            distribution_vector = self.distribution_vector.squeeze(1)
            wei_1, wei_2 = self.mem_w1(distribution_vector), self.mem_w2(distribution_vector)
            bias_1, bias_2 = self.mem_b1(distribution_vector), self.mem_b2(distribution_vector)
            self.mlp_layers[1].memory_parameters(wei_1, bias_1)
            self.mlp_layers[3].memory_parameters(wei_2, bias_2)

        scores = self.mlp_layers(seq_emb).view(candidate_items_emb.shape[0], -1)

        return scores  # [batch_size, NEG_ITEMS_NUM+1]

    def _get_candidates_emb(self, item_id=None):
        # item_id: (batch_size, n_inline_items)
        candidate_items_emb = self.item_embedding(item_id)  # [batch_size, NEG_ITEMS_NUM+1, 64]
        return candidate_items_emb

    def _distribution_vector_etractor(self, candidate_items_emb):
        distri_vec = self.extract_distribution_layer(candidate_items_emb)  # [batch_size, 1, 64]
        if len(distri_vec.size()) == 2:
            distri_vec = distri_vec.unsqueeze(1)
        return distri_vec  #.reshape((-1, self.embedding_size))

    def _input_modulation(self, input_tensor, distribution_vector):
        if len(input_tensor.size()) == 2:
            input_tensor = input_tensor.unsqueeze(1)  # [batch_size, 1, embedding_size]

        gamma = self.film_affine_emb_scale(distribution_vector)  # [batch_size, 1, 1]
        beta = self.film_affine_emb_bias(distribution_vector)  # [batch_size, 1, 1]
        output_tensor = gamma * input_tensor + beta

        return output_tensor  # [batch_size, 1, embedding_size]
