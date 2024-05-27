import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1) #x 4,17,24,,,, x_mark 4,24,3 ####   x:4,24,17 _> 4,17,24
        # x: [Batch Variate Time]
        # if x_mark is None:
        x = self.value_embedding(x)  # x 4,17,512
        # else:
        #     x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) #
        # x: [Batch Variate d_model]
        return self.dropout(x)




class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1] #n_vars :17
        x = self.padding_patch_layer(x) #x (4,17,10)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # 4，17，3，16
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))# 68,3,16
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) #68,3,32
        return self.dropout(x), n_vars
