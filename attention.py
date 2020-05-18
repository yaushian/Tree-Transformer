import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from modules import *

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None, group_prob=None, no_cuda=False):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        seq_len=query.size()[-2]
        if no_cuda:
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32),0))
        else:
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0)).cuda()
        scores = scores.masked_fill((mask|b) == 0, -1e9)
    if group_prob is not None:
        p_attn = F.softmax(scores, dim = -1)
        p_attn = p_attn*group_prob.unsqueeze(1)
    else:
        p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, no_cuda=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.no_cuda = no_cuda
        
    def forward(self, query, key, value, group_prob=None, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # query,key,value shape: (nbatches, h, seq_len, d_k)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout, group_prob=group_prob, no_cuda=self.no_cuda)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class GroupAttention(nn.Module):
    def __init__(self, d_model, dropout=0.8, no_cuda=False):
        super(GroupAttention, self).__init__()
        self.d_model = 256.
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        #self.linear_output = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.no_cuda = no_cuda

    def forward(self, context, eos_mask, prior):
        batch_size,seq_len = context.size()[:2]

        context =self.norm(context)

        if self.no_cuda:
            a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),1))
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32),0))
            c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),-1))
            tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len,seq_len], dtype=np.float32),0))
        else:
            a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),1)).cuda()
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32),0)).cuda()
            c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),-1)).cuda()
            tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len,seq_len], dtype=np.float32),0)).cuda()

        #mask = eos_mask & (a+c) | b
        mask = eos_mask & (a+c)
        
        key = self.linear_key(context)
        query = self.linear_query(context)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model
        
        scores = scores.masked_fill(mask == 0, -1e9)
        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn*neibor_attn.transpose(-2,-1) + 1e-9)
        neibor_attn = prior + (1. - prior)*neibor_attn

        t = torch.log(neibor_attn + 1e-9).masked_fill(a==0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int()-b)==0, 0)     
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b==0, 1e-9)
        
        return g_attn,neibor_attn
