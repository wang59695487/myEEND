# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ModuleList

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

from typing import Optional

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class ResTransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
        * add short-cut connection between input & output of individual Encoder Block
        * output all-block results

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(ResTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        outputs = []
        for mod in self.layers:
            output += mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask) # learning the residual info...
            outputs.append(output)

        if self.norm is not None:
            outputs = [self.norm(output) for output in outputs]

        return outputs

class NoamScheduler(_LRScheduler):
    """
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Args:
        d_model: int
            The number of expected features in the encoder inputs.
        warmup_steps: int
            The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

        # the initial learning rate is set as step = 1
        if self.last_epoch == -1:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
            self.last_epoch = 0
        print(self.d_model)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class TransformerModel(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_pos=False, residual=False):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerModel, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        if not residual:
            self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        else:
            self.transformer_encoder = ResTransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(n_units, n_speakers)
        self.residual = residual
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, ilens=None, has_mask=False, activation=None):
        if ilens is not None:
            orilen = src.shape[1]
            maxlen = int(ilens.max().cpu().item())
            src = src[:,:maxlen,:]
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != srz.size(1):
                mask = self._generate_square_subsequent_mask(srz.size(1)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True) # padding对齐并且把list变成tensor

        # src: (B, T, E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        output = self.transformer_encoder(src, self.src_mask)

        # if residual, we got multiple outputs from different encoders
        if self.residual and self.training:
            opts = []
            for opt in output:
                opt = opt.transpose(0, 1)
                # output: (B, T, C)
                opt = self.decoder(opt)
                if activation:
                    opt = activation(opt)
                # padding back to origin seqence length, otherwise error when gather ther results from different gpu
                if ilens is not None:
                    opt = F.pad(opt, [0, 0, 0, orilen-maxlen, 0, 0])
                opts.append(opt)

            return opts
        if self.residual:
            output = output[-1]
        # output: (B, T, E)
        output = output.transpose(0, 1)
        # output: (B, T, C)
        output = self.decoder(output)

        if activation:
            output = activation(output)

        # padding back to origin seqence length, otherwise error when gather ther results from different gpu
        if ilens is not None:
            output = F.pad(output, [0, 0, 0, orilen-maxlen, 0, 0])

        #output = [out[:ilen] for out, ilen in zip(output, ilens)]

        return output

    def get_attention_weight(self, src):
        # NOTE: NOT IMPLEMENTED CORRECTLY!!!
        attn_weight = []
        def hook(module, input, output):
            # attn_output, attn_output_weights = multihead_attn(query, key, value)
            # output[1] are the attention weights
            attn_weight.append(output[1])
            
        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))

        self.eval()
        with torch.no_grad():
            self.forward(src)

        for handle in handles:
            handle.remove()
        self.train()

        return torch.stack(attn_weight)


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional information to each time step of x
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    import torch
    model = TransformerModel(5, 40, 4, 512, 2, dropout=0.1)
    input = torch.randn(8, 500, 40)
    print("Model output:", model(input)[0].size())
    #print("Model attention:", model.get_attention_weight(input).size())
    #print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())
