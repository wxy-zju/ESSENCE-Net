import math
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Linear, LayerNorm


class Encoder(nn.Module):
    def __init__(self, c_in):
        super(Encoder, self).__init__()
        self.encoder_norm = LayerNorm(c_in, eps=1e-6)
        self.layer = Block(c_in)

    def forward(self, hidden_states):
        hidden_states = self.layer(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded



class Attention(nn.Module):
    def __init__(self, c_in):
        super(Attention, self).__init__()
        self.num_attention_heads = 4
        self.attention_head_size = int(c_in / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(c_in, self.all_head_size)
        self.key = Linear(c_in, self.all_head_size)
        self.value = Linear(c_in, self.all_head_size)
        self.out = Linear(c_in, c_in)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        return attention_output


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Mlp(nn.Module):
    def __init__(self, c_in):
        super(Mlp, self).__init__()
        self.fc1 = Linear(c_in, c_in)
        self.fc2 = Linear(c_in, c_in)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, c_in):
        super(Block, self).__init__()
        self.hidden_size = c_in
        self.attention_norm = LayerNorm(c_in, eps=1e-6)
        self.ffn_norm = LayerNorm(c_in, eps=1e-6)
        self.ffn = Mlp(c_in)
        self.attn = Attention(c_in)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

class Transformer(nn.Module):
    def __init__(self, c_in):
        super(Transformer, self).__init__()
        self.encoder = Encoder(c_in)

    def forward(self,x):
        input_shape = x.shape
        input = rearrange(x, 'b c n h w -> (b h w) n c')
        out = self.encoder(input)
        out = rearrange(out, '(b h w) n c -> b c n h w',b =input_shape[0],h=input_shape[3],w=input_shape[4])
        return out

