import torch.nn as nn

from .seqrec_base import SeqRecBase


class GRU4Rec(SeqRecBase):    
    def __init__(self, config):
        super(GRU4Rec, self).__init__(config)

    def _define_model_layers(self):
        # gru
        self.hidden_size = self.config['hidden_size']
        self.num_layers = 1
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)  

    def forward_user_emb(self, user_id=None, item_seq=None, item_seq_len=None, item_seq_features=None, time_seq=None):
        item_seq_emb = self.item_embedding_for_user(item_seq, item_seq_features, time_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb)
        gru_output = self.dense(gru_output)

        seq_output = gru_output[:, -1]

        return seq_output


     