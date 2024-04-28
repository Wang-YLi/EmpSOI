### TAKEN FROM https://github.com/kolloldas/torchnlp
import os
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
from torch.nn import init
from itertools import repeat
from torch_geometric.nn import SAGEConv,to_hetero,GATConv
import torch_geometric.transforms as T
import dgl.nn.pytorch as dglnn
import dgl
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from statistics import mean
# from LibMTL.weighting import AbsWeighting


import numpy as np
import math
from model.common import (
    EncoderLayer,
    DecoderLayer,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    NoamOpt,
    LayerNorm,
    LabelSmoothing,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
)
from model import config


MAP_EMO = {
    0: "surprised",
    1: "excited",
    2: "annoyed",
    3: "proud",
    4: "angry",
    5: "sad",
    6: "grateful",
    7: "lonely",
    8: "impressed",
    9: "afraid",
    10: "disgusted",
    11: "confident",
    12: "terrified",
    13: "hopeful",
    14: "anxious",
    15: "disappointed",
    16: "joyful",
    17: "prepared",
    18: "guilty",
    19: "furious",
    20: "nostalgic",
    21: "jealous",
    22: "anticipating",
    23: "embarrassed",
    24: "content",
    25: "devastated",
    26: "sentimental",
    27: "caring",
    28: "trusting",
    29: "ashamed",
    30: "apprehensive",
    31: "faithful",
}

from sklearn.metrics import accuracy_score


def find_true(tensor):
    true_index = []
    for item in tensor:
        index = []
        for i in range(len(item)):
            if(item[i]==True):
                index.append(i)
        true_index.append(index)
    return true_index

class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=10000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        src_mask, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (src_mask, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (src_mask, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq
            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_num = 4 if config.woEMO else 5
        input_dim = input_num * config.hidden_dim
        hid_num = 2 if config.woEMO else 3
        hid_dim = hid_num * config.hidden_dim
        out_dim = config.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x


class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1])   # 默认沿着中间所有的shape
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        if config.hidden_dim % config.gat_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_dim, config.gat_heads))
        self.config = config
        self.num_attention_heads = config.gat_heads
        self.attention_head_size = int(config.hidden_dim / config.gat_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_dim, self.all_head_size)
        self.key = nn.Linear(config.hidden_dim, self.all_head_size)
        self.value = nn.Linear(config.hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # Normalize the attention scores to probabilities.
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand_as(attention_scores)
            attention_probs = nn.Softmax(dim=-1)(attention_scores.masked_fill(attention_mask==0, -1e9))
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data),
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (
            (kernel_size - 1, 0)
            if pad_type == "left"
            else (kernel_size // 2, (kernel_size - 1) // 2)
        )
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(
            input_size, output_size, kernel_size=kernel_size, padding=0
        )

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs
    
class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(
        self,
        input_depth,
        filter_size,
        output_depth,
        layer_config="ll",
        padding="left",
        dropout=0.1,
    ):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data),
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = (
            [(input_depth, filter_size)]
            + [(filter_size, filter_size)] * (len(layer_config) - 2)
            + [(filter_size, output_depth)]
        )

        for lc, s in zip(list(layer_config), sizes):
            if lc == "l":
                layers.append(nn.Linear(*s))
            elif lc == "c":
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x
    
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(
        self,
        input_depth,
        total_key_depth,
        total_value_depth,
        output_depth,
        num_heads,
        bias_mask=None,
        dropout=0.1,
    ):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

        if total_key_depth % num_heads != 0:
            print(
                "Key depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_key_depth, num_heads)
            )
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print(
                "Value depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_value_depth, num_heads)
            )
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(
            shape[0], shape[1], self.num_heads, shape[2] // self.num_heads
        ).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(shape[0], shape[2], shape[3] * self.num_heads)
        )

    def forward(self, queries, keys, values, mask):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        ## attention weights
        attetion_weights = logits.sum(dim=1) / self.num_heads

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        return outputs, attetion_weights
    
class MaskedSelfAttention(nn.Module):
    def __init__(self, config):
        super(MaskedSelfAttention, self).__init__()
        if config.hidden_dim % config.gat_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_dim, config.gat_heads))
        self.config = config
        self.num_attention_heads = config.gat_heads
        self.attention_head_size = int(config.hidden_dim / config.gat_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_dim, self.all_head_size)
        self.key = nn.Linear(config.hidden_dim, self.all_head_size)
        self.value = nn.Linear(config.hidden_dim, self.all_head_size)

        self.output = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.graph_layer_norm = LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self.fusion = nn.Linear(2*config.hidden_dim, config.hidden_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, role_mask, conv_len, state_type='user'):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        if role_mask is not None:
            role_mask = role_mask.unsqueeze(1).expand_as(attention_scores)
            attention_probs = nn.Softmax(dim=-1)(attention_scores.masked_fill(role_mask==0, -1e9))
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.output(context_layer)
        context_layer = self.graph_layer_norm(context_layer + hidden_states)
        high_emo_states = []
        perspective_taking = []
        for item, idx in zip(context_layer, conv_len):
            if state_type == 'other':
                perspective_taking.append(item[6*idx].unsqueeze(0))
                high_emo_states.append(item[6*idx+1].unsqueeze(0))
            else:
                perspective_taking.append(item[6*idx+2].unsqueeze(0))
                high_emo_states.append(item[6*idx+3].unsqueeze(0))
        high_emo_states = torch.cat(high_emo_states, dim=0)
        perspective_taking = torch.cat(perspective_taking, dim=0)

        return context_layer, high_emo_states, perspective_taking

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.lin_1 = nn.Linear(5*config.hidden_dim, 5*config.hidden_dim, bias=False)
        self.lin_2 = nn.Linear(5*config.hidden_dim, config.hidden_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x


class PGD():
    def __init__(self, embedding):
        self.embedding = embedding
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='embedding', is_first_attack=False):
        name = 'embedding'
        param = self.embedding.lut.weight
        if is_first_attack:
            self.emb_backup[name] = param.data.clone()
        norm = torch.norm(param.grad)
        if norm != 0:
            r_at = alpha * param.grad / norm
            param.data.add_(r_at)
            param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='embedding'):
        name = 'embedding'
        param = self.embedding.lut.weight
        assert name in self.emb_backup
        param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        name = 'embedding'
        param = self.embedding.lut.weight
        if param.requires_grad:
            self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        name = 'embedding'
        param = self.embedding.lut.weight
        if param.requires_grad:
            param.grad = self.grad_backup[name]


class EmpSOI(nn.Module):
    def __init__(
        self,
        vocab,
        emo_number,
        model_file_path=None,
        is_eval=False,
        load_optim=False,
        
    ):
        super(EmpSOI, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.word_freq = np.zeros(self.vocab_size)

        self.is_eval = is_eval
        self.rels = ["x_intent", "x_attr", "x_want", "x_effect","x_need", "x_react"]

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)

        self.dropout = SpatialDropout(0.3)
        self.lin = nn.Linear(config.hidden_dim*3, config.hidden_dim)
        self.encoder = self.make_encoder(config.emb_dim)
        self.ref_encoder = self.make_encoder(config.emb_dim*2)
        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )
        self.pgd = PGD(self.embedding)

        self.emo_lin = nn.Linear(2*config.hidden_dim, emo_number,bias=False)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        
        self.emo_garrix = nn.Linear(emo_number, config.hidden_dim, bias=False)
        self.layer_norm = LayerNorm(config.hidden_dim)
        self.cog_cross_attn = CrossAttention(config)
        self.cross_attn = CrossAttention(config)
        self.graph_cross_attn = CrossAttention(config)
        self.all_cross_attn = CrossAttention(config)
        self.fusion = PositionwiseFeedForward(3*config.hidden_dim, 3*config.hidden_dim,config.hidden_dim)
        self.graph_ref_fusion = PositionwiseFeedForward(2*config.hidden_dim, 2*config.hidden_dim, config.hidden_dim)
        self.cog_ref_fusion = PositionwiseFeedForward(6*config.hidden_dim,6*config.hidden_dim,config.hidden_dim)
        self.reg_fusion = nn.Linear(2*config.hidden_dim, config.hidden_dim)
        self.cog_multihead = MultiHeadAttention(config.hidden_dim,config.hidden_dim,config.hidden_dim,config.hidden_dim,4) 
        self.graph_multihead = MultiHeadAttention(config.hidden_dim,config.hidden_dim,config.hidden_dim,config.hidden_dim,4)
        self.multihead = MultiHeadAttention(config.hidden_dim,config.hidden_dim,config.hidden_dim,config.hidden_dim,4) 


        self.self_imagine_interaction = MaskedSelfAttention(config)
        self.other_imagine_interaction = MaskedSelfAttention(config)

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.activation = nn.Softmax(dim=1)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx, reduction="sum")
        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.criterion_ppl_ls =LabelSmoothing(size=vocab.n_words,padding_idx=config.PAD_idx,smoothing=0.1)                                               
        self.sigmoid=nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def make_encoder(self, emb_dim):
        return Encoder(
            emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

    def save_model(self, running_avg_ppl, iter):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "EmpISO_{}_{:.4f}".format(iter, running_avg_ppl),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def clean_preds(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if config.EOS_idx in pred:
                ind = pred.index(config.EOS_idx) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == config.SOS_idx:
                pred = pred[1:]
            res.append(pred)
        return res

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)
        for k, v in curr.items():
            if k != config.EOS_idx:
                self.word_freq[k] += v

    def calc_weight(self):
        RF = self.word_freq / self.word_freq.sum()
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)

        return torch.FloatTensor(weight).to(config.device)
    def construct_graph(self, utt_hidden,utt_cls_index,conv_len ,x_attr,commonsense):
        aware_graph = []
        ##初始化节点
        self_perspective_taking = torch.zeros([utt_hidden.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_hidden).cuda()
        other_perspective_taking = x_attr.unsqueeze(1).type_as(utt_hidden).cuda()
        self_high_emo_state = torch.rand([utt_hidden.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_hidden).cuda()
        other_high_emo_state = torch.rand([utt_hidden.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_hidden).cuda()
        utt_cls_embs =  []
        for item, idx in zip(utt_hidden, utt_cls_index):
            cls_emb = torch.index_select(item, 0, idx)
            utt_cls_embs.append(cls_emb)
        utt_cls_embs = torch.stack(utt_cls_embs, dim=0)
        for i, cur_len in enumerate(conv_len):
            utt = utt_cls_embs[i, :cur_len]
            for j in range(cur_len):
                for k in range(5):
                    utt = torch.cat([utt, commonsense[k][i][j].unsqueeze(0)], dim=0)
            utt = torch.cat([utt.cuda(), other_perspective_taking[i].cuda(), other_high_emo_state[i].cuda(), self_perspective_taking[i].cuda(), self_high_emo_state[i].cuda()], dim=0).cuda()
            aware_graph.append(utt)
        aware_graph = pad_sequence(aware_graph, batch_first=True, padding_value=config.PAD_idx)
        return aware_graph
    def forward(self, batch):
        ## Encode the context (Semantic Knowledge)
        enc_batch = batch["input_batch"]
        batch_size = enc_batch.shape[0]
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1) # 判断是否是PAD，输出True/False
        mask_emb = self.embedding(batch["mask_input"]) # dialogue state
        src_emb = self.dropout(self.embedding(enc_batch))+ mask_emb
        enc_outputs = self.encoder(src_emb, src_mask)  # batch_size * seq_len * 300
        conv_len = [len(item) for item in batch["input_txt"]]
        utt_cls_index = batch["x_cls_index"]
        last_index =batch["last_user_cls_index"]
        # Commonsense relations
        cs_outputs_graph = []
        cog_logit = []
        for r in self.rels:
            if(r!='x_attr'):
                    cls_mask = batch[r].data.eq(config.CLS_idx).to(config.device)
                    true_index = find_true(cls_mask)
                    emb = self.embedding(batch[r].long()).to(config.device)
                    mask = batch[r].data.eq(config.PAD_idx).unsqueeze(1)
                    enc_output = self.encoder(emb, mask)
                    cls_embs =  []
                    for item, idx in zip(enc_output, true_index):
                        cls_emb = torch.index_select(item, 0, torch.LongTensor(idx).to(config.device)).to(config.device)
                        cls_embs.append(cls_emb)
                    cs_outputs_graph.append(cls_embs)
                    cog_cls = [torch.mean(torch.Tensor(item),dim=0) for item in cls_embs]
                    cog_cls = torch.stack(cog_cls,dim=0)
                    cog_logit.append(cog_cls)
                    if(r == 'x_react'):
                        emo_cls = [torch.mean(torch.Tensor(item),dim=0) for item in cls_embs]
                        emo_cls = torch.stack(emo_cls,dim=0)
            else:
                emb = self.embedding(batch[r].long()).to(config.device)
                mask = batch[r].data.eq(config.PAD_idx).unsqueeze(1)
                enc_output = self.encoder(emb, mask)
                perspective_taking = torch.mean(enc_output,dim=1)
        # graph
        self_other_graph = self.construct_graph(enc_outputs,utt_cls_index ,conv_len ,perspective_taking,cs_outputs_graph).to(config.device)
        x, other_high_emo_state, other_perspective_taking = self.other_imagine_interaction(self_other_graph, batch["other_mask"], conv_len, 'other')
        x_other, other_high_emo_state, other_perspective_taking = self.other_imagine_interaction(x, batch["other_mask"], conv_len, 'other')

        x, self_high_emo_state, self_perspective_taking = self.self_imagine_interaction(self_other_graph, batch["self_mask"], conv_len, 'self')
        x_self, self_high_emo_state, self_perspective_taking = self.self_imagine_interaction(x, batch["self_mask"], conv_len, 'self')

        all_graph_result = self.cross_attn(other_high_emo_state.unsqueeze(1),self_high_emo_state.unsqueeze(1),self_high_emo_state.unsqueeze(1))
        # Shape: batch_size * 1 * 300
        # Emotions
        last_cls_embs =  []
        for i in range(batch_size):
            cls_emb = enc_outputs[i][last_index[i]]
            last_cls_embs.append(cls_emb)
        last_cls_embs = torch.stack(last_cls_embs, dim=0)
        emo_logits = self.emo_lin(torch.concat([emo_cls.squeeze(1),last_cls_embs],dim=1))

        cog_logit = torch.tensor([item.cpu().detach().numpy() for item in cog_logit]).cuda()
        cog_logit = cog_logit.view(cog_logit.size(1),config.hidden_dim*5)

        # Cognition
        cog_ref_ctx = self.cog_ref_fusion(torch.cat([enc_outputs, cog_logit.unsqueeze(1).expand(enc_outputs.size(0),enc_outputs.size(1),1500)], dim=-1))
        cog_ref_ctx,_ = self.cog_multihead(cog_ref_ctx,cog_ref_ctx, cog_ref_ctx,src_mask)
        
        graph_ref_ctx= self.graph_ref_fusion(torch.cat([enc_outputs, all_graph_result.expand_as(enc_outputs)], dim=-1))
        graph_ref_ctx,_ = self.graph_multihead(graph_ref_ctx, graph_ref_ctx, graph_ref_ctx,src_mask)
        
        enc_outputs = self.all_cross_attn(enc_outputs,cog_ref_ctx,cog_ref_ctx)+enc_outputs
        return enc_outputs, src_mask, emo_logits,other_perspective_taking,self_perspective_taking
    def train_one_batch(self, batch, iter, train=True):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        enc_outputs,src_mask ,emo_logits,other_graph_result,self_graph_result= self.forward(batch)
        
        # Decode
        sos_token = (
            torch.LongTensor([config.SOS_idx] * enc_batch.size(0))
            .unsqueeze(1)
            .to(config.device)
        )

        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        # batch_size * seq_len * 300 (GloVe)
        dec_emb = self.embedding(dec_batch_shift)
        pre_logit, attn_dist = self.decoder(dec_emb, enc_outputs, (src_mask, mask_trg))

        # gate = nn.Sigmoid()(self.fusion(torch.cat([pre_logit, other_graph_result.unsqueeze(1).expand_as(pre_logit), self_graph_result.unsqueeze(1).expand_as(pre_logit)], dim=-1)))
        # pre_logit = pre_logit + gate * other_graph_result.unsqueeze(1).expand_as(pre_logit) + (1-gate) * self_graph_result.unsqueeze(1).expand_as(pre_logit)


        ## compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )

        emo_label = torch.LongTensor(batch["program_label"]).to(config.device)
        emo_loss = nn.CrossEntropyLoss()(emo_logits, emo_label).to(config.device)
        # emo_loss = self.criterion_emo(emo_logits, emo_label).to(config.device)
        ctx_loss = self.criterion_ppl(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )
        ctx_loss_ls = self.criterion_ppl_ls(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )
        _, preds = logit.max(dim=-1)
        preds = self.clean_preds(preds)
        self.update_frequency(preds)
        self.criterion.weight = self.calc_weight()
        not_pad = dec_batch.ne(config.PAD_idx)
        target_tokens = not_pad.long().sum().item()
        div_loss = self.criterion(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )
        div_loss /= target_tokens
        # loss = emo_loss*emo_loss  +   ctx_loss*ctx_loss + div_loss*div_loss
        loss = emo_loss  +   ctx_loss + div_loss*1.5

        

        pred_program = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        # print results for testing
        top_preds = ""
        comet_res = {}

        if self.is_eval:
            top_preds = emo_logits.detach().cpu().numpy().argsort()[0][-3:][::-1]
            top_preds = f"{', '.join([MAP_EMO[pred.item()] for pred in top_preds])}"
            # for r in self.rels:
            #     txt = [[" ".join(t) for t in tm] for tm in batch[f"{r}_txt"]][0]
            #     comet_res[r] = txt

        if train:
            loss.backward()
            self.optimizer.step()

        return (
            ctx_loss.item(),
            math.exp(min(ctx_loss.item(), 100)),
            emo_loss.item(),
            program_acc,
            top_preds,
            comet_res,
        )

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        (
            _,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        # src_mask, ctx_output, _ = self.forward(batch)
        enc_outputs,src_mask ,emo_logits,other_graph_result,self_graph_result= self.forward(batch)
        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(enc_outputs),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    ys_embed, enc_outputs, (src_mask, mask_trg)
                )
            # gate = nn.Sigmoid()(self.fusion(torch.cat([out, other_graph_result.unsqueeze(1).expand_as(out), self_graph_result.unsqueeze(1).expand_as(out)], dim=-1)))
            # out = out + gate * other_graph_result.unsqueeze(1).expand_as(out) + (1-gate) * self_graph_result.unsqueeze(1).expand_as(out)
            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ ,_,int_T= self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), ctx_output, (src_mask, mask_trg)
                )
            dim = [-1, out.size(1), -1]
            out = self.layer_norm(int_T.expand(dim)+out)
            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logit, dim=-1)

            next_word = torch.multinomial(probs, 1).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            # _, next_word = torch.max(logit[:, -1], dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent