import torch
import torch.nn as nn
import copy
from .seqrec_base import SeqRecBase
import math

ACT2FN = {
        "gelu": nn.functional.gelu, 
        "relu": nn.functional.relu, 
        "swish": nn.functional.silu}


class ConvFormer(SeqRecBase):
    """ A novel sequential model. 
    For model details, please refer to https://arxiv.org/abs/2308.02925.
    """   
    def __init__(self, config):
        self.conv_size = config['conv_size']
        self.padding_mode = config['padding_mode']
        self.n_layers = config['n_layers']
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = float(config['layer_norm_eps'])
        self.max_seq_len = config['max_seq_len']
        self.seq_decay = config['seq_decay']
        self.seq_merge = config['seq_merge']
        self.init_ratio = config['init_ratio']
        if self.conv_size > self.max_seq_len:
            raise ValueError(f"`conv_size` should be smaller than `max_seq_len`, while get `conv={self.conv_size}` and `max_seq_len={self.max_seq_len}`.")
        super(ConvFormer, self).__init__(config)

    def _define_model_layers(self): 
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        layer = Layer(
                conv_size=self.conv_size, 
                padding_mode=self.padding_mode, 
                hidden_dropout_prob=self.hidden_dropout_prob, 
                hidden_size=self.hidden_size, 
                inner_size=self.inner_size, 
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                init_ratio=self.init_ratio)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.n_layers)])
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)  
    
    def forward_user_emb(self, user_id=None, item_seq=None, item_seq_len=None, item_seq_features=None, time_seq=None):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding_for_user(item_seq, item_seq_features, time_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        hidden_states = self.dropout(input_emb)
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states)
        if self.seq_merge:
            non_zero_num = item_seq_len.unsqueeze(-1) + 1
            output = (hidden_states * torch.logspace(self.seq_decay, 0, steps=self.max_seq_len).unsqueeze(0).unsqueeze(-1).to(hidden_states.device)).sum(1) / non_zero_num.pow(0.5)
        else:
            output = hidden_states[:, -1, :]
        return output  # [B H]


class ConvFormerLayer(nn.Module):
    def __init__(self, conv_size, padding_mode, hidden_dropout_prob, hidden_size, layer_norm_eps, init_ratio):
        super(ConvFormerLayer, self).__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.conv = nn.Sequential()
        self.padding_len = conv_size - 1
        self.init_ratio = init_ratio
        if padding_mode not in ['circular', 'reflect', 'constant']:
            raise ValueError(f"`padding_mode` has three optional values: ['circular', 'reflect', 'constant'], while got {padding_mode}")
        self.padding_mode = padding_mode

        conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=conv_size, groups=hidden_size)
        conv.weight.data.normal_(0.0, self.init_ratio)
        conv.bias.data.normal_(0.0, self.init_ratio)
        self.conv.add_module('depthwise_conv', conv)

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        x = input_tensor.transpose(1, 2)
        if self.padding_mode == 'circular':
            x = torch.cat((x[:, :, -self.padding_len:], x), dim=2)
        elif self.padding_mode == 'reflect':
            x = torch.cat((torch.flip(x, dims=[2])[:,:,0:self.padding_len], x), dim=2)
        elif self.padding_mode == 'constant':
            x = torch.cat((torch.zeros((x.size(0), x.size(1), self.padding_len)).to(x.device), x), dim=2)
        x = self.conv(x).transpose(1, 2)
        hidden_states = self.out_dropout(x)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_act, hidden_dropout_prob, layer_norm_eps):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = ACT2FN[hidden_act]
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states) 
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):
    def __init__(self, conv_size, padding_mode, hidden_dropout_prob, hidden_size, inner_size, hidden_act, layer_norm_eps, init_ratio):
        super(Layer, self).__init__()

        self.filterlayer = ConvFormerLayer(conv_size, padding_mode, hidden_dropout_prob, hidden_size, layer_norm_eps, init_ratio)
        self.intermediate = Intermediate(hidden_size, inner_size, hidden_act, hidden_dropout_prob, layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.filterlayer(hidden_states)
        output = self.intermediate(hidden_states)
        return output