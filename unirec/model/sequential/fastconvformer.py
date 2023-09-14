import torch
import torch.nn as nn
import copy
from .convformer import ConvFormer, Intermediate

class Layer(nn.Module):
    def __init__(self, conv_size, padding_mode, hidden_dropout_prob, hidden_size, inner_size, hidden_act, layer_norm_eps, max_seq_len):
        super(Layer, self).__init__()

        self.filterlayer = FASTConvFormerLayer(conv_size, padding_mode, hidden_dropout_prob, hidden_size, layer_norm_eps, max_seq_len)
        self.intermediate = Intermediate(hidden_size, inner_size, hidden_act, hidden_dropout_prob, layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.filterlayer(hidden_states)
        output = self.intermediate(hidden_states)
        return output

class FASTConvFormerLayer(nn.Module):
    def __init__(self, conv_size, padding_mode, hidden_dropout_prob, hidden_size, layer_norm_eps, max_seq_len):
        super(FASTConvFormerLayer, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(1, conv_size, hidden_size, dtype=torch.float32) * 0.02)
        self.zeros = nn.Parameter(torch.zeros(1, max_seq_len-conv_size, hidden_size), requires_grad=False)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.conv = nn.Sequential()
        self.padding_len = conv_size - 1

        # In Spectral Convolution, the default padding mode for conv kernel is zero-padding, 
        # which makes the result theoretically consistent to the ConvFormer approach with the hidden states zeroly padded.
        # Padding should be conducted on the right, since fconv is equal to the vanilla convolution with flip.
        if padding_mode == 0:
            self.padding_mode = 'circular'
        elif padding_mode == 1:
            self.padding_mode = 'reflect'
        elif padding_mode == 2:
            self.padding_mode = 'constant'

        conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=conv_size, groups=hidden_size)
        init_ratio = 5e-3
        conv.weight.data.normal_(0.0, init_ratio)
        conv.bias.data.normal_(0.0, init_ratio)
        self.conv.add_module('depthwise_conv', conv)

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        # Implementation of Spectral Convolution
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.cat([self.conv_weight, self.zeros], dim=1) # padding the kernel to the equal length with input
        weight = torch.fft.rfft(weight, dim=1, norm='ortho') # conduct FFT on the kernel
        hidden_states = x * weight # convolution in the fourier domain
        hidden_states = torch.fft.irfft(hidden_states, n=seq_len, dim=1, norm='ortho') # back to the temporal domain
        
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FASTConvFormer(ConvFormer):
    """ A faster version of ConvFormer. 
    For model details, please refer to https://arxiv.org/abs/2308.02925.
    """
    def _define_model_layers(self): 
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        layer = Layer(
                conv_size=self.conv_size, 
                padding_mode=self.padding_mode, 
                hidden_dropout_prob=self.hidden_dropout_prob, 
                hidden_size=self.hidden_size, 
                inner_size=self.inner_size, 
                hidden_act='gelu',
                layer_norm_eps=self.layer_norm_eps, 
                max_seq_len=self.max_seq_len
                )
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.n_layers)])
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
