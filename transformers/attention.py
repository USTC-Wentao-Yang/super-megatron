# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : attention.py
# Time       ：2025/4/3 20:17
# Author     ：Wentao Yang
"""
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        self.WQ = nn.Linear(embed_dim, embed_dim)
        self.WK = nn.Linear(embed_dim, embed_dim)
        self.WV = nn.Linear(embed_dim, embed_dim)
        self.Dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        attention_scores = torch.softmax(attention_scores, -1)
        output = torch.matmul(attention_scores, V)
        return output


def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 1

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    bs = 512
    seq_len = 100
    embed_dim = 10

    set_seed(42)
    x = torch.rand(bs, seq_len, embed_dim).cuda()

    mask_true = create_causal_mask(seq_len)
    print("mask")
    print(mask_true)
    print("-" * 50)

    self_atten = SelfAttention(embed_dim)
    self_atten = self_atten.cuda()
    res = self_atten(x)
    print(res)

    x = torch.ones(2, 3, 5)
    y = torch.full((5,), 10)
    print(x * y)