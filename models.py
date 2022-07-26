#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/4 9:16
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : benchmark.py
# @Description : In this script, the implementation of BERT refers from https://github.com/dhlee347/pytorchic-bert

from torch.nn.modules.conv import Conv2d
from utils import split_last, merge_last

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_pretrain import RadarAE_prototype,PretrainVisionTransformerEncoderMine
from functools import partial
from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        # return x

class LayerNormMine(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg['hidden']), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg['hidden']), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        # return x


class Embeddings(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)

class Embeddings_mine_v1(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        # x: (N, T, R, C)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)
        # N,T,R,C = x.shape
        # factorized embedding
        # x = x.view(N,T,R*C)
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)

class ConvBatchPool(nn.Module):
    def __init__(self, cin, cout, size, strides=(1,1), padding=(0,0), poolSize=(2,2)):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, size, strides, padding)
        self.batchNorm = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()
        if poolSize != (1,1) and poolSize != [1,1]:
            self.pool = nn.MaxPool2d(poolSize)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class ConvTransBatch(nn.Module):
    def __init__(self, cin, cout, size, strides=(1,1), padding=(0,0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin, cout, size, strides, padding)
        self.batchNorm = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x

class LinearBatchReLU(nn.Module):
    def __init__(self, featureIn, featureOut):
        super().__init__()
        self.linear1 = nn.Linear(featureIn, featureOut)
        self.batchNorm = nn.BatchNorm1d(featureOut)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x

class Embeddings_mine_v2(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()
        cin = cfg.channel
        # factorized embedding
        self.conv1 = ConvBatchPool(cin, 16, (5,5), (1,1), padding=(2,2), poolSize=(1,2))
        self.conv2 = ConvBatchPool(16, 32, (5,5), (1,1), padding=(2,2), poolSize=(1,2))
        self.conv3 = ConvBatchPool(32, 64, (5,5), (1,1), padding=(2,2), poolSize=(1,2))
        self.conv4 = ConvBatchPool(64, 64, (5,3), (1,1), padding=(2,1), poolSize=(1,1))
        self.dense1 = nn.Linear(24//8*64,cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        # x: (N, T, R, C)
        N, T, F = x.shape
        x = x.view(N,T,F//2,2)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)
        # factorized embedding
        # e = self.lin(x)
        e = x.permute(0,3,1,2)
        e = self.conv1(e)
        e = self.conv2(e)
        e = self.conv3(e)
        e = self.conv4(e)
        N,C,H,W = e.shape
        e = e.permute(0,2,3,1)
        e = e.reshape(N,H,W*C)
        e = self.dense1(e)

        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)

class EmbeddingsMineV1(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(cfg['feature_num'], cfg['hidden'])
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg['input_shape'][0], cfg['hidden']) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNormMine(cfg)
        self.emb_norm = cfg['emb_norm']

    def forward(self, x):
        # x: (N, T, R, C)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)
        # N,T,R,C = x.shape
        # factorized embedding
        # x = x.view(N,T,R*C)
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)

class EmbeddingsMineV2(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()
        cin = cfg['channel']
        self.conv1 = ConvBatchPool(cin, 16, (5,5), (1,1), padding=(2,2), poolSize=(1,2))# 71x20x2->71x10x16
        self.conv2 = ConvBatchPool(16, 32, (5,5), (1,1), padding=(2,2), poolSize=(1,2))# 71x10x16->71x5x32
        self.conv3 = ConvBatchPool(32, 64, (5,5), (1,1), padding=(2,2), poolSize=(1,1))# 71x5x32->71x5x64
        self.conv4 = ConvBatchPool(64, 128, (5,3), (1,1), padding=(2,1), poolSize=(1,1))# 71x5x64->71x5x128
        self.dense1 = nn.Linear(5*128,cfg['hidden'])
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg['input_shape'][0], cfg['hidden']) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNormMine(cfg)
        self.emb_norm = cfg['emb_norm']

    def forward(self, x):
        # x: (N, T, R, C)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)
        e = x.permute(0,3,1,2)
        e = self.conv1(e)
        e = self.conv2(e)
        e = self.conv3(e)
        e = self.conv4(e)
        N,C,H,W = e.shape
        e = e.permute(0,2,3,1)
        e = e.reshape(N,H,W*C)
        e = self.dense1(e)

        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)

class EmbeddingsMineV3(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()
        cin = cfg['channel']
        self.conv1 = ConvBatchPool(cin, 16, (5,5), (1,1), padding=(2,2), poolSize=(1,2))# 71x20x2->71x10x16
        self.conv2 = ConvBatchPool(16, 32, (5,5), (1,1), padding=(2,2), poolSize=(1,2))# 71x10x16->71x5x32
        self.conv3 = ConvBatchPool(32, 64, (5,5), (1,1), padding=(2,2), poolSize=(1,1))# 71x5x32->71x5x64
        self.conv4 = ConvBatchPool(64, 128, (5,3), (1,1), padding=(2,1), poolSize=(1,1))# 71x5x64->71x5x128
        self.dense1 = nn.Linear(5*128,cfg['hidden'])
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg['input_shape'][0], cfg['hidden']) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNormMine(cfg)
        self.emb_norm = cfg['emb_norm']

    def forward(self, x):
        # x: (N, T, R, C)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)
        e = x.permute(0,3,1,2)
        e = self.conv1(e)
        e = self.conv2(e)
        e = self.conv3(e)
        e = self.conv4(e)
        N,C,H,W = e.shape
        e = e.permute(0,2,3,1)
        e = e.reshape(N,H,W*C)
        e = self.dense1(e)

        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)

class Encoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        cin = cfg.channel
        self.conv1 = ConvBatchPool(cin, 32, (5,5), (1,1), padding=(2,2), poolSize=(2,2))
        self.conv2 = ConvBatchPool(32, 64, (5,5), (1,1), padding=(2,2), poolSize=(1,1))
        self.conv3 = ConvBatchPool(64, 128, (5,5), (1,1), padding=(2,2), poolSize=(1,1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        cin = cfg.channel
        self.conv1 = ConvTransBatch(128, 64, 5, strides=1, padding=2)
        self.conv2 = ConvTransBatch(64, 32, 5, strides=1, padding=2)
        self.conv3 = ConvTransBatch(32, cin, 4, strides=2, padding=(1,1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class MultiProjection(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        return q, k, v


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        #scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))

class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        h = self.embed(x)

        for _ in range(self.n_layers):
            # h = block(h, mask)
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h

class MultiHeadAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)

    def forward(self,h):
        h = self.attn(h)
        h = self.norm1(h + self.proj(h))
        h = self.norm2(h + self.pwff(h))
        return h

class Transformer_mine(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings_mine_v2(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.Layers = nn.Sequential(*[MultiHeadAttn(cfg) for _ in range(self.n_layers)])
        # self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first= True, bidirectional=True)
        self.dense = nn.Linear(2*cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        h = self.embed(x)

        h = self.Layers(h)
        # h = self.gru(h)[0]
        # h = self.dense(h)
        return h

class Transformer_mineV2(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings_mine_v2(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.Layers = nn.Sequential(*[MultiHeadAttn(cfg) for _ in range(self.n_layers)])
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first= True, bidirectional=True)
        self.dense = nn.Linear(2*cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        h = self.embed(x)

        h = self.Layers(h)
        h = self.gru(h)[0]
        h = self.dense(h)
        return h

class Transformer_mineV3(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings_mine_v2(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.Layers = nn.TransformerEncoderLayer(cfg.hidden, cfg.n_heads, dim_feedforward=cfg.hidden_ff, dropout=0, activation='relu')
        self.n_layers = cfg.n_layers
        # self.Layers = nn.Sequential(*[MultiHeadAttn(cfg) for _ in range(self.n_layers)])
        # self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first= True, bidirectional=True)
        # self.dense = nn.Linear(2*cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        y = self.embed(x)
        y = y.transpose(0,1)
        y = self.Layers(y)
        y = y.transpose(0,1)
        # h = self.gru(h)[0]
        # h = self.dense(h)
        return y

class Transformer_mineV4(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings_mine_v2(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.Layers = nn.Sequential(*[nn.TransformerEncoderLayer(cfg.hidden, cfg.n_heads, dim_feedforward=cfg.hidden_ff, dropout=0, activation='relu') for _ in range(cfg.n_layers)])
        self.n_layers = cfg.n_layers
        # self.Layers = nn.Sequential(*[MultiHeadAttn(cfg) for _ in range(self.n_layers)])
        # self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first= True, bidirectional=True)
        # self.dense = nn.Linear(2*cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        y = self.embed(x)
        y = y.transpose(0,1)
        y = self.Layers(y)
        y = y.transpose(0,1)
        # h = self.gru(h)[0]
        # h = self.dense(h)
        return y

class Transformer_mineV5(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings_mine_v1(cfg)

        self.Layers = nn.Sequential(*[nn.TransformerEncoderLayer(cfg.hidden, cfg.n_heads, dim_feedforward=cfg.hidden_ff, dropout=0, activation='gelu') for _ in range(cfg.n_layers)])
        self.n_layers = cfg.n_layers
        # self.Layers = nn.Sequential(*[MultiHeadAttn(cfg) for _ in range(self.n_layers)])
        # self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first= True, bidirectional=True)
        # self.dense = nn.Linear(2*cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        y = self.embed(x)
        y = y.transpose(0,1)
        y = self.Layers(y)
        y = y.transpose(0,1)
        # h = self.gru(h)[0]
        # h = self.dense(h)
        return y

class TransformerMineV1(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = EmbeddingsMineV2(cfg)
        self.Layers = nn.Sequential(*[nn.TransformerEncoderLayer(cfg['hidden'], cfg['n_heads'], dim_feedforward=cfg['hidden_ff'], dropout=0, activation='gelu') for _ in range(cfg['n_layers'])])

    def forward(self, x):
        y = self.embed(x)
        y = y.transpose(0,1)
        y = self.Layers(y)
        y = y.transpose(0,1)
        return y

class TransformerMineV2(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = EmbeddingsMineV2(cfg)
        self.gru = nn.GRU(cfg['hidden'], cfg['hidden'], num_layers=cfg['n_layers'], batch_first= True, bidirectional=True)
        self.dense = nn.Linear(2*cfg['hidden'], cfg['hidden'])
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        y = self.embed(x)
        y = self.gru(y)[0]
        y = self.dense(y)
        return y

class LIMUBertModel4Pretrain(nn.Module):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer(cfg) # encoder
        # self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        # self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        # self.activ = gelu
        # self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        # h_masked = self.activ(self.linear(h_masked))
        # h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm

class LIMUBertModel4Pretrain_mine(nn.Module):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer_mine(cfg) # encoder
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm

class LIMUBertModel4Pretrain_mineV2(nn.Module):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer_mineV2(cfg) # encoder
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm

class LIMUBertModel4Pretrain_mineV3(nn.Module):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer_mineV3(cfg) # encoder
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm

class LIMUBertModel4Pretrain_mineV4(nn.Module):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer_mineV4(cfg) # encoder
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm

class PretrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ouput_embed = False

class LIMUBertModel4Pretrain_mineV5(PretrainModel):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer_mineV5(cfg) # encoder
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm

def buildEncoder(cfg, output_embed=False, model_file=None):
    name = cfg['type']
    model:nn.Module = eval(name)(cfg, output_embed=output_embed)
    if model_file:
        print("loading weights from {}".format(model_file + '.pt'))
        model.load_state_dict(torch.load(model_file + '.pt'))
        model.requires_grad = False
    return model

def buildDecoder(cfg, model_file=None):
    name = cfg['type']
    model = eval(name)(cfg)
    if model_file:
        print("loading weights from {}".format(model_file + '.pt'))
        model.load_state_dict(torch.load(model_file + '.pt'))
        model.requires_grad = False
    return model


class PreTrainModelV1(PretrainModel):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = TransformerMineV1(cfg) # encoder
        self.linear = nn.Linear(cfg['hidden'], cfg['hidden'])
        self.activ = gelu
        self.norm = LayerNormMine(cfg)
        self.decoder = nn.Linear(cfg['hidden'], math.prod(cfg["input_shape"][1:]))
        self.input_shape = cfg["input_shape"]
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        N, T = logits_lm.shape[:2]
        logits_lm = logits_lm.view(N,T,*self.input_shape[1:])
        return logits_lm

class PreTrainModelV2(PretrainModel):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = TransformerMineV2(cfg) # encoder
        self.linear = nn.Linear(cfg['hidden'], cfg['hidden'])
        self.activ = gelu
        self.norm = LayerNormMine(cfg)
        self.decoder = nn.Linear(cfg['hidden'], math.prod(cfg["input_shape"][1:]))
        self.input_shape = cfg["input_shape"]
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        N, T = logits_lm.shape[:2]
        logits_lm = logits_lm.view(N,T,*self.input_shape[1:])
        return logits_lm

class PreTrainModelV3(PretrainModel):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = TransformerMineV2(cfg) # encoder
        self.decoder = nn.Linear(cfg['hidden'], math.prod(cfg["input_shape"][1:]))
        self.input_shape = cfg["input_shape"]
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        logits_lm = self.decoder(h_masked)
        N, T = logits_lm.shape[:2]
        logits_lm = logits_lm.view(N,T,*self.input_shape[1:])
        return logits_lm

class PreTrainModelV4(PretrainModel):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = TransformerMineV1(cfg) # encoder
        # self.linear = nn.Linear(cfg['hidden'], cfg['hidden'])
        # self.activ = gelu
        # self.norm = LayerNormMine(cfg)
        self.decoder = nn.Linear(cfg['hidden'], math.prod(cfg["input_shape"][1:]))
        self.input_shape = cfg["input_shape"]
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        # h_masked = self.activ(self.linear(h_masked))
        # h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        N, T = logits_lm.shape[:2]
        logits_lm = logits_lm.view(N,T,*self.input_shape[1:])
        return logits_lm

PRETRAIN = 0
CLASSIFY = 0

class CompositeModel(nn.Module):

    def __init__(self, encoder:PretrainModel, decoder:nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.ouput_embed = True
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(True)

    def forward(self, input):
        embbeded = self.encoder(input)
        output = self.decoder(embbeded)
        return output

class ClassifierLSTM(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('lstm' + str(i), nn.LSTM(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i], batch_first=True))
            else:
                self.__setattr__('lstm' + str(i),
                                 nn.LSTM(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i],
                                         batch_first=True))
            self.__setattr__('bn' + str(i), nn.BatchNorm1d(cfg.seq_len))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            lstm = self.__getattr__('lstm' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h, _ = lstm(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierGRU(nn.Module):
    def __init__(self, cfg, input=None, output=None, feats=False):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('gru' + str(i), nn.GRU(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i], batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i],
                                         batch_first=True))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierAttn(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.embd = nn.Embedding(cfg.seq_len, input)
        self.proj_q = nn.Linear(input, cfg.atten_hidden)
        self.proj_k = nn.Linear(input, cfg.atten_hidden)
        self.proj_v = nn.Linear(input, cfg.atten_hidden)
        self.attn = nn.MultiheadAttention(cfg.atten_hidden, cfg.num_head)
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.flatten = nn.Flatten()
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        seq_len = input_seqs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device)
        pos = pos.unsqueeze(0).expand(input_seqs.size(0), seq_len)  # (S,) -> (B, S)
        h = input_seqs + self.embd(pos)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        h, weights = self.attn(q, k, v)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            if i == self.num_linear - 1:
                h = self.flatten(h)
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierCNN2D(nn.Module):
    def __init__(self, cfg, output=None):
        super().__init__()
        for i in range(cfg.num_cnn):
            if i == 0:
                self.__setattr__('cnn' + str(i), nn.Conv2d(1, cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            else:
                self.__setattr__('cnn' + str(i), nn.Conv2d(cfg.conv_io[i][0], cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            self.__setattr__('bn' + str(i), nn.BatchNorm2d(cfg.conv_io[i][1]))
        self.pool = nn.MaxPool2d(cfg.pool[0], stride=cfg.pool[1], padding=cfg.pool[2])
        self.flatten = nn.Flatten()
        for i in range(cfg.num_linear):
            if i == 0:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.flat_num, cfg.linear_io[i][1]))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_cnn = cfg.num_cnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs.unsqueeze(1)
        for i in range(self.num_cnn):
            cnn = self.__getattr__('cnn' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h = cnn(h)
            if self.activ:
                h = F.relu(h)
            h = bn(self.pool(h))
            # h = self.pool(h)
        h = self.flatten(h)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierCNN1D(nn.Module):
    def __init__(self, cfg, output=None):
        super().__init__()
        for i in range(cfg.num_cnn):
            if i == 0:
                self.__setattr__('cnn' + str(i),
                                 nn.Conv1d(cfg.seq_len, cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            else:
                self.__setattr__('cnn' + str(i),
                                 nn.Conv1d(cfg.conv_io[i][0], cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            self.__setattr__('bn' + str(i), nn.BatchNorm1d(cfg.conv_io[i][1]))
        self.pool = nn.MaxPool1d(cfg.pool[0], stride=cfg.pool[1], padding=cfg.pool[2])
        self.flatten = nn.Flatten()
        for i in range(cfg.num_linear):
            if i == 0:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.flat_num, cfg.linear_io[i][1]))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_cnn = cfg.num_cnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_cnn):
            cnn = self.__getattr__('cnn' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h = cnn(h)
            if self.activ:
                h = F.relu(h)
            h = self.pool(h)
            # h = bn(h)
            # h = self.pool(h)
        h = self.flatten(h)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class BERTClassifier(nn.Module):

    def __init__(self, bert_cfg, classifier=None, frozen_bert=False):
        super().__init__()
        self.transformer = Transformer(bert_cfg)
        if frozen_bert:
            for p in self.transformer.parameters():
                p.requires_grad = False
        self.classifier = classifier

    def forward(self, input_seqs, training=False): #, training
        h = self.transformer(input_seqs)
        h = self.classifier(h, training)
        return h

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)


class LIMUBenchmarkDCNN(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, (5, 1))
        self.bn1 = nn.BatchNorm2d(50)
        self.conv2 = nn.Conv2d(50, 40, (5, 1))
        self.bn2 = nn.BatchNorm2d(40)
        if cfg.seq_len <= 20:
            self.conv3 = nn.Conv2d(40, 20, (2, 1))
        else:
            self.conv3 = nn.Conv2d(40, 20, (3, 1))
        self.bn3 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d((2, 1))
        self.lin1 = nn.Linear(input * cfg.flat_num, 400)
        self.lin2 = nn.Linear(400, output)

    def forward(self, input_seqs, training=False):
        h = input_seqs.unsqueeze(1)
        h = F.relu(F.tanh(self.conv1(h)))
        h = self.bn1(self.pool(h))
        h = F.relu(F.tanh(self.conv2(h)))
        h = self.bn2(self.pool(h))
        h = F.relu(F.tanh(self.conv3(h)))
        h = h.view(h.size(0), h.size(1), h.size(2) * h.size(3))
        h = self.lin1(h)
        h = F.relu(F.tanh(torch.sum(h, dim=1)))
        h = self.normalize(h[:, :, None, None])
        h = self.lin2(h[:, :, 0, 0])
        return h

    def normalize(self, x, k=1, alpha=2e-4, beta=0.75):
        # x = x.view(x.size(0), x.size(1) // 5, 5, x.size(2), x.size(3))#
        # y = x.clone()
        # for s in range(x.size(0)):
        #     for j in range(x.size(1)):
        #         for i in range(5):
        #             norm = alpha * torch.sum(torch.square(y[s, j, i, :, :])) + k
        #             norm = torch.pow(norm, -beta)
        #             x[s, j, i, :, :] = y[s, j, i, :, :] * norm
        # x = x.view(x.size(0), x.size(1) * 5, x.size(3), x.size(4))
        return x


class BenchmarkDeepSense(nn.Module):

    def __init__(self, cfg, input=None, output=None, num_filter=8):
        super().__init__()
        self.sensor_num = input // 3
        for i in range(self.sensor_num):
            self.__setattr__('conv' + str(i) + "_1", nn.Conv2d(1, num_filter, (2, 3)))
            self.__setattr__('conv' + str(i) + "_2", nn.Conv2d(num_filter, num_filter, (3, 1)))
            self.__setattr__('conv' + str(i) + "_3", nn.Conv2d(num_filter, num_filter, (2, 1)))
            self.__setattr__('bn' + str(i) + "_1", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_2", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_3", nn.BatchNorm2d(num_filter))
        self.conv1 = nn.Conv2d(1, num_filter, (2, self.sensor_num))
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, (3, 1))
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = nn.Conv2d(num_filter, num_filter, (2, 1))
        self.bn3 = nn.BatchNorm2d(num_filter)
        self.flatten = nn.Flatten()

        self.lin1 = nn.Linear(cfg.flat_num, 12)
        self.lin2 = nn.Linear(12, output)


    def forward(self, input_seqs, training=False):
        h = input_seqs.view(input_seqs.size(0), input_seqs.size(1), self.sensor_num, 3)
        hs = []
        for i in range(self.sensor_num):
            t = h[:, :, i, :]
            t = torch.unsqueeze(t, 1)
            for j in range(3):
                cv = self.__getattr__('conv' + str(i) + "_" + str(j + 1))
                bn = self.__getattr__('bn' + str(i) + "_" + str(j + 1))
                t = bn(F.relu(cv(t)))
            hs.append(self.flatten(t)[:, :, None])
        h = torch.cat(hs, dim=2)
        h = h.unsqueeze(1)
        h = self.bn1(F.relu(self.conv1(h)))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        h = self.flatten(h)
        h = self.lin2(F.relu(self.lin1(h)))
        return h


class BenchmarkTPNPretrain(nn.Module):
    def __init__(self, cfg, task_num, input=None):
        super().__init__()
        self.conv1 = nn.Conv1d(input, 32, kernel_size=6)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=2)
        self.flatten = nn.Flatten()
        for i in range(task_num):
            self.__setattr__('slin' + str(i) + "_1", nn.Linear(96, 256))
            self.__setattr__('slin' + str(i) + "_2", nn.Linear(256, 1))
        self.task_num = task_num

    def forward(self, input_seqs, training=False):
        h = input_seqs.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv2(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv3(h))
        h = F.dropout(h, p=0.1, training=training)
        h = self.flatten(torch.max(h, 2)[0])
        hs = []
        for i in range(self.task_num):
            lin1 = self.__getattr__('slin' + str(i) + "_1")
            lin2 = self.__getattr__('slin' + str(i) + "_2")
            hl = F.relu(lin1(h))
            hl = F.sigmoid(lin2(hl))
            hs.append(hl)
        hf = torch.stack(hs)[:, :, 0]
        hf = torch.transpose(hf, 0, 1)
        return hf


class BenchmarkTPNClassifier(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.conv1 = nn.Conv1d(input, 32, kernel_size=6)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(96, 1024)
        self.fc2 = nn.Linear(1024, output)
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.conv2.parameters():
            p.requires_grad = False
        for p in self.conv3.parameters():
            p.requires_grad = False

    def forward(self, input_seqs, training=False):
        h = input_seqs.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv2(h))
        h = F.dropout(h, p=0.1, training=training)
        h = F.relu(self.conv3(h))
        h = F.dropout(h, p=0.1, training=training)
        h = self.flatten(torch.max(h, 2)[0])
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return h

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class CNNv3(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.fc1 = LinearBatchReLU(60*1024,32)
        self.fc2 = nn.Linear(32, 7)
        self.flatten = nn.Flatten()

    def forward(self, input_seqs, training=False):
        self.train(training)
        N, T, F = input_seqs.shape
        x = self.flatten(input_seqs)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class Dense(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()

        self.fc1 = LinearBatchReLU(math.prod(cfg['input_shape']),32)
        self.fc2 = nn.Linear(32, 7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)


class Gru(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.gru = nn.GRU(cfg['input_shape'][1],256,2,batch_first=True,bidirectional=True)
        self.fc1 = LinearBatchReLU(cfg['input_shape'][0]*256*2,32)
        self.fc2 = nn.Linear(32, 7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.gru(x)[0]
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class Dense2(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()

        self.fc1 = LinearBatchReLU(math.prod(cfg['input_shape']),512)
        self.fc2 = nn.Linear(512, 7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class DecoderTransformer(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(cfg['input_shape'][1],6,4*cfg['input_shape'][1])
        self.fc1 = LinearBatchReLU(math.prod(cfg['input_shape']),512)
        self.fc2 = nn.Linear(512, 7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.permute(1,0,2)
        x = self.transformer(x).permute(1,0,2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class DecoderTransformer4(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(cfg['input_shape'][1],cfg['n_heads'],cfg['mlp_ratio']*cfg['input_shape'][1])
        self.fc1 = LinearBatchReLU(math.prod(cfg['input_shape']),512)
        self.fc2 = nn.Linear(512, 7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.permute(1,0,2)
        x = self.transformer(x).permute(1,0,2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class DecoderTransformer2(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(cfg['input_shape'][1],6,4*cfg['input_shape'][1])
        self.fc1 = LinearBatchReLU(math.prod(cfg['input_shape']),512)
        self.fc2 = nn.Linear(512, 7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = self.transformer(x).permute(1,0,2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class DecoderTransformer3(nn.Module):

    def __init__(self, cfg:dict, input=None, output=None):
        super().__init__()
        shareWeight = cfg.get("shareWeight",False)
        if not shareWeight:
            self.transformers = nn.ModuleList([nn.TransformerEncoderLayer(cfg['input_shape'][1],6,4*cfg['input_shape'][1]) for _ in range(cfg['n_layers'])])
        else:
            block = nn.TransformerEncoderLayer(cfg['input_shape'][1],6,4*cfg['input_shape'][1])
            self.transformers = nn.ModuleList([block for _ in range(cfg['n_layers'])])
        self.fc1 = LinearBatchReLU(math.prod(cfg['input_shape']),512)
        self.fc2 = nn.Linear(512, 7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.permute(1,0,2)
        for block in self.transformers:
            x = block(x)
        x = x.permute(1,0,2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class DecoderTransformer4(nn.Module):

    def __init__(self, cfg:dict, input=None, output=None):
        super().__init__()
        shareWeight = cfg.get("shareWeight",False)
        if not shareWeight:
            self.transformers = nn.ModuleList([nn.TransformerEncoderLayer(cfg['input_shape'][1],6,4*cfg['input_shape'][1]) for _ in range(cfg['n_layers'])])
        else:
            block = nn.TransformerEncoderLayer(cfg['input_shape'][1],6,4*cfg['input_shape'][1])
            self.transformers = nn.ModuleList([block for _ in range(cfg['n_layers'])])
        self.fc1 = LinearBatchReLU(math.prod(cfg['input_shape']),7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.permute(1,0,2)
        for block in self.transformers:
            x = block(x)
        x = x.permute(1,0,2)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class DecoderTransformer5(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.transformers = nn.ModuleList([nn.TransformerEncoderLayer(cfg['input_shape'][1],6,4*cfg['input_shape'][1]) for _ in range(cfg['n_layers'])])
        self.fc1 = LinearBatchReLU(math.prod(cfg['input_shape']),512)
        self.fc2 = nn.Linear(512, 7)
        self.flatten = nn.Flatten()
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg['input_shape'][0], cfg['input_shape'][1]))

    def forward(self, x):
        x = x+ self.pos_embed.type_as(x).to(x.device).clone().detach()
        x = x.permute(1,0,2)
        for block in self.transformers:
            x = block(x)
        x = x.permute(1,0,2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class DecoderTransformer6(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.transformers = nn.ModuleList([nn.TransformerEncoderLayer(cfg['input_shape'][1],6,4*cfg['input_shape'][1]) for _ in range(cfg['n_layers'])])
        self.fc1 = LinearBatchReLU(math.prod(cfg['input_shape']),512)
        self.fc2 = nn.Linear(512, 7)
        self.flatten = nn.Flatten()
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg['input_shape'][0], cfg['input_shape'][1]))

    def forward(self, x):
        x = x+ self.pos_embed.to(x.device)
        x = x.permute(1,0,2)
        for block in self.transformers:
            x = block(x)
        x = x.permute(1,0,2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class GRU_DENSE(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.rnn = nn.GRU(cfg['input_shape'][1],256,2,batch_first=True,bidirectional=True)
        self.fc1 = LinearBatchReLU(512*cfg['input_shape'][0],32)
        self.fc2 = nn.Linear(32, 7)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.rnn(x)[0]
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class CNN_GRU(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.emb = Embeddings_mine_v2(cfg)
        self.fc1 = LinearBatchReLU(60*1024,32)
        self.fc2 = nn.Linear(32, 7)
        self.flatten = nn.Flatten()
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first= True, bidirectional=True)
        self.dense = nn.Linear(2*cfg.hidden, cfg.hidden)

    def forward(self, input_seqs, training=False):
        self.train(training)
        x = self.emb(input_seqs)
        x = self.gru(x)[0]
        x = self.dense(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class BenchmarkDCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.conv1 = ConvBatchPool(2, 16, (5,5), (1,1), padding=(2,2), poolSize=(1,2))# 71x20x2->71x10x16
        self.conv2 = ConvBatchPool(16, 32, (5,5), (1,1), padding=(2,2), poolSize=(1,2))# 71x10x16->71x5x32
        self.conv3 = ConvBatchPool(32, 64, (5,5), (1,1), padding=(2,2), poolSize=(1,1))# 71x5x32->71x5x64
        self.conv4 = ConvBatchPool(64, 128, (5,3), (1,1), padding=(2,1), poolSize=(1,1))# 71x5x64->71x5x128
        self.dense1 = nn.Linear(71*5*128,7)

    def forward(self, x):
        # x: (N, T, R, C)
        e = x.permute(0,3,1,2)
        e = self.conv1(e)
        e = self.conv2(e)
        e = self.conv3(e)
        e = self.conv4(e)
        N,C,H,W = e.shape
        e = e.permute(0,2,3,1)
        e = e.reshape(N,H*W*C)
        e = self.dense1(e)
        return e

class BenchmarkDCNN2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.conv1 = ConvBatchPool(2, 16, (5,5), (1,1), padding=(2,2), poolSize=(1,2))# 71x20x2->71x10x16
        self.conv2 = ConvBatchPool(16, 32, (5,5), (1,1), padding=(2,2), poolSize=(1,2))# 71x10x16->71x5x32
        self.conv3 = ConvBatchPool(32, 64, (5,5), (1,1), padding=(2,2), poolSize=(1,1))# 71x5x32->71x5x64
        self.dense1 = nn.Linear(71*5*64,7)

    def forward(self, x):
        # x: (N, T, R, C)
        e = x.permute(0,3,1,2)
        e = self.conv1(e)
        e = self.conv2(e)
        e = self.conv3(e)
        N,C,H,W = e.shape
        e = e.permute(0,2,3,1)
        e = e.reshape(N,H*W*C)
        e = self.dense1(e)
        return e

class BenchmarkDCNN3(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.conv1 = ConvBatchPool(2, 16, (5,5), (1,1), padding=(2,2), poolSize=(2,2))# 72x20x2->36x10x16
        self.conv2 = ConvBatchPool(16, 32, (5,5), (1,1), padding=(2,2), poolSize=(2,2))# 36x10x16->18x5x32
        self.conv3 = ConvBatchPool(32, 64, (5,5), (1,1), padding=(2,2), poolSize=(2,1))# 18x5x32->9x5x64
        self.dense1 = LinearBatchReLU(9*5*64,1024)
        self.dense2 = nn.Linear(1024,7)

    def forward(self, x):
        # x: (N, T, R, C)
        e = x.permute(0,3,1,2)
        e = self.conv1(e)
        e = self.conv2(e)
        e = self.conv3(e)
        N,C,H,W = e.shape
        e = e.permute(0,2,3,1)
        e = e.reshape(N,H*W*C)
        e = self.dense1(e)
        e = self.dense2(e)
        return e

class BenchmarkSVM(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.dense1 = nn.Linear(72*20*2,7)

    def forward(self, x):
        # x: (N, T, R, C)
        N,T,R,C = e.shape

        e = e.reshape(N,T*R*C)
        e = self.dense1(e)
        return e

class BenchmarkSoli(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.svm = nn.Linear(6*72,7)

    def forward(self, x):
        # x: (N, T, R, C)
        ans = self.svm(x)
        return ans

class BenchmarkDCNNGRU(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.conv1 = ConvBatchPool(2, 16, (5,5), (1,1), padding=(2,2), poolSize=(2,2))# 72x20x2->36x10x16
        self.conv2 = ConvBatchPool(16, 32, (5,5), (1,1), padding=(2,2), poolSize=(2,2))# 36x10x16->18x5x32
        self.conv3 = ConvBatchPool(32, 64, (5,5), (1,1), padding=(2,2), poolSize=(2,1))# 18x5x32->9x5x64
        self.conv4 = ConvBatchPool(64, 128, (5,3), (1,1), padding=(2,1), poolSize=(1,1))# 8x5x64->8x5x128
        self.rnn = nn.GRU(5*128,128,2,batch_first=True,bidirectional=True)
        self.dense1 = LinearBatchReLU(9*2*128,7)

    def forward(self, x):
        # x: (N, T, R, C)
        e = x.permute(0,3,1,2)
        e = self.conv1(e)
        e = self.conv2(e)
        e = self.conv3(e)
        N,C,H,W = e.shape
        e = torch.flatten(e,2)
        e = self.rnn(e)
        e = torch.flatten(e,1)
        e = self.dense1(e)
        return e

class BenchmarkMMCL(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.ModuleList([
            ConvBatchPool(1, 4, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x32x20x1->71x16x10x4
            ConvBatchPool(4, 8, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x16x10x4->71x8x5x8
            ConvBatchPool(8, 16, (5,5), (1,1), padding=(2,2), poolSize=(2,1)),# 71x8x5x8->71x4x5x16
            ConvBatchPool(16, 16, (5,3), (1,1), padding=(2,1), poolSize=(1,1))# 71x4x5x16->71x4x5x16
        ])
        self.conv2 = nn.ModuleList([
            ConvBatchPool(1, 4, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x8x20x1->71x4x10x4
            ConvBatchPool(4, 8, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x4x10x4->71x2x5x8
            ConvBatchPool(8, 16, (5,5), (1,1), padding=(2,2), poolSize=(2,1)),# 71x2x5x8->71x1x5x16
            ConvBatchPool(16, 16, (5,3), (1,1), padding=(2,1), poolSize=(1,1))# 71x1x5x16->71x1x5x16
        ])
        self.rnn = nn.GRU(5*5*16,128,2,batch_first=True,bidirectional=True)
        self.dense1 = nn.Linear(71*256,7)

    def forward(self, x1, x2):
        # x: (N, T, R, C)
        N,T,H1,W = x1.shape
        N,T,H2,W = x2.shape
        x1 = x1.reshape(N*T,1,H1,W)
        x2 = x2.reshape(N*T,1,H2,W)
        for block in self.conv1:
            x1 = block(x1)
        for block in self.conv2:
            x2 = block(x2)
        x1 = x1.reshape(N,T,-1)
        x2 = x2.reshape(N,T,-1)
        x = torch.cat([x1,x2],-1)
        x = self.rnn(x)[0]
        x = torch.flatten(x,1)
        x = self.dense1(x)
        return x

class BenchmarkRFWash(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.ModuleList([
            ConvBatchPool(1, 4, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x32x20x1->71x16x10x4
            ConvBatchPool(4, 8, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x16x10x4->71x8x5x8
            ConvBatchPool(8, 16, (5,5), (1,1), padding=(2,2), poolSize=(2,1)),# 71x8x5x8->71x4x5x16
            ConvBatchPool(16, 16, (5,3), (1,1), padding=(2,1), poolSize=(1,1))# 71x4x5x16->71x4x5x16
        ])
        self.rnn = nn.GRU(4*5*16,128,2,batch_first=True,bidirectional=True)
        self.dense1 = nn.Linear(71*256,7)

    def forward(self, x1, x2):
        # x: (N, T, R, C)
        N,T,H1,W = x1.shape
        x1 = x1.reshape(N*T,1,H1,W)
        for block in self.conv1:
            x1 = block(x1)
        x1 = x1.reshape(N,T,-1)
        x = self.rnn(x1)[0]
        x = torch.flatten(x,1)
        x = self.dense1(x)
        return x

class BenchmarkRFWash_VA(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.ModuleList([
            ConvBatchPool(1, 4, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x32x20x1->71x16x10x4
            ConvBatchPool(4, 8, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x16x10x4->71x8x5x8
            ConvBatchPool(8, 16, (5,5), (1,1), padding=(2,2), poolSize=(2,1)),# 71x8x5x8->71x4x5x16
            ConvBatchPool(16, 16, (5,3), (1,1), padding=(2,1), poolSize=(1,1))# 71x4x5x16->71x4x5x16
        ])
        
        self.conv2 = nn.ModuleList([ #10 22
            ConvBatchPool(1, 4, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 5 11
            ConvBatchPool(4, 8, (5,5), (1,1), padding=(2,2), poolSize=(1,2)),# 5 5
            ConvBatchPool(8, 16, (5,5), (1,1), padding=(2,2), poolSize=(1,1)),
            ConvBatchPool(16, 16, (5,3), (1,1), padding=(2,1), poolSize=(1,1))
        ])
        self.rnn = nn.GRU((5*5+4*5)*16,128,2,batch_first=True,bidirectional=True)
        self.dense1 = nn.Linear(71*256,7)

    def forward(self, x1, x2):
        # x: (N, T, R, C)
        N,T,H1,W = x1.shape
        N,T,H2,W = x2.shape
        x1 = x1.reshape(N*T,1,H1,W)
        x2 = x2.reshape(N*T,1,H2,W)
        for block in self.conv1:
            x1 = block(x1)
        for block in self.conv2:
            x2 = block(x2)
        x1 = x1.reshape(N,T,-1)
        x2 = x2.reshape(N,T,-1)
        x = torch.cat([x1,x2], -1)
        x = self.rnn(x)[0]
        x = torch.flatten(x,1)
        x = self.dense1(x)
        return x

class BenchmarkDeepSoli(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.ModuleList([
            ConvBatchPool(1, 4, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x32x20x1->71x16x10x4
            ConvBatchPool(4, 8, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x16x10x4->71x8x5x8
            ConvBatchPool(8, 16, (5,5), (1,1), padding=(2,2), poolSize=(2,1)),# 71x8x5x8->71x4x5x16
            ConvBatchPool(16, 16, (5,3), (1,1), padding=(2,1), poolSize=(1,1))# 71x4x5x16->71x4x5x16
        ])
        self.rnn = nn.LSTM(4*5*16,128,2,batch_first=True,bidirectional=False)
        self.dense1 = nn.Linear(128,7)

    def forward(self, x1, x2):
        # x: (N, T, R, C)
        N,T,H1,W = x1.shape
        x1 = x1.reshape(N*T,1,H1,W)
        for block in self.conv1:
            x1 = block(x1)
        x1 = x1.reshape(N,T,-1)
        x = self.rnn(x1)[0]
        x = self.dense1(x)
        x = x.mean(1)
        return x

class BenchmarkDeepSoli_VA(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.ModuleList([
            ConvBatchPool(1, 4, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x32x20x1->71x16x10x4
            ConvBatchPool(4, 8, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 71x16x10x4->71x8x5x8
            ConvBatchPool(8, 16, (5,5), (1,1), padding=(2,2), poolSize=(2,1)),# 71x8x5x8->71x4x5x16
            ConvBatchPool(16, 16, (5,3), (1,1), padding=(2,1), poolSize=(1,1))# 71x4x5x16->71x4x5x16
        ])
        self.conv2 = nn.ModuleList([ #10 22
            ConvBatchPool(1, 4, (5,5), (1,1), padding=(2,2), poolSize=(2,2)),# 5 11
            ConvBatchPool(4, 8, (5,5), (1,1), padding=(2,2), poolSize=(1,2)),# 5 5
            ConvBatchPool(8, 16, (5,5), (1,1), padding=(2,2), poolSize=(1,1)),
            ConvBatchPool(16, 16, (5,3), (1,1), padding=(2,1), poolSize=(1,1))
        ])
        self.rnn = nn.LSTM((5*5+4*5)*16,128,2,batch_first=True,bidirectional=False)
        self.dense1 = nn.Linear(128,7)

    def forward(self, x1, x2):
        N,T,H1,W = x1.shape
        N,T,H2,W = x2.shape
        x1 = x1.reshape(N*T,1,H1,W)
        x2 = x2.reshape(N*T,1,H2,W)
        for block in self.conv1:
            x1 = block(x1)
        for block in self.conv2:
            x2 = block(x2)
        x1 = x1.reshape(N,T,-1)
        x2 = x2.reshape(N,T,-1)
        x = torch.cat([x1,x2],-1)
        x = self.rnn(x)[0]
        x = self.dense1(x)
        x = x.mean(1)
        return x

class BenchmarkDeepGRU(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.gru = nn.GRU(20*2,256,2,batch_first=True,bidirectional=True)
        self.dense1 = LinearBatchReLU(72*256*2,1024)
        self.dense2 = nn.Linear(1024,7)

    def forward(self, x):
        # x: (N, T, R, C)
        x = torch.flatten(x,2)
        e = self.gru(x)[0]
        e = torch.flatten(e,1)
        e = self.dense1(e)
        e = self.dense2(e)
        return e

class BenchmarkDeepBiLSTM(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.gru = nn.LSTM(20*2,256,2,batch_first=True,bidirectional=True)
        self.dense1 = LinearBatchReLU(72*256*2,1024)
        self.dense2 = nn.Linear(1024,7)

    def forward(self, x):
        # x: (N, T, R, C)
        x = torch.flatten(x,2)
        e = self.gru(x)[0]
        e = torch.flatten(e,1)
        e = self.dense1(e)
        e = self.dense2(e)
        return e

class BenchmarkRadarAE_noPretrain(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        encoder_cfg = {
            "type": "RadarAE",
            "channel" : 2,
            "input_shape" : [72,20,2],
            "patch_size" : [2,10],
            "hidden": 300,
            "hidden_ff": 256,
            "mlp_ratio":4,
            "n_layers": 3,
            "n_heads": 12,
            "emb_norm": False,
            "decoderType":"TFC",
            "hidden_decoder":150,
            "n_layers_decoder":1,
            "n_heads_decoder":3
        }
        decoder_cfg = {
            "type":"DecoderTransformer3",
            "input_shape":[72,300],
            "n_layers":2
        }
        self.encoder = buildEncoder(encoder_cfg,output_embed=True,model_file=None)
        self.decoder = buildDecoder(decoder_cfg)
        self.encoder.ouput_embed = True

    def forward(self, input):
        embbeded = self.encoder(input)
        output = self.decoder(embbeded)
        return output

        

def fetch_classifier(method, model_cfg, input=None, output=None, feats=False):
    if 'lstm' in method:
        model = ClassifierLSTM(model_cfg, input=input, output=output)
    elif 'gru' in method:
        model = ClassifierGRU(model_cfg, input=input, output=output)
    elif 'dcnn' in method:
        model = BenchmarkDCNN(model_cfg, input=input, output=output)
    elif 'cnn2' in method:
        model = ClassifierCNN2D(model_cfg, output=output)
    elif 'cnn1' in method:
        model = ClassifierCNN1D(model_cfg, output=output)
    elif 'deepsense' in method:
        model = BenchmarkDeepSense(model_cfg, input=input, output=output)
    elif 'attn' in method:
        model = ClassifierAttn(model_cfg, input=input, output=output)
    elif 'cnn3' in method:
        model = CNNv3(model_cfg)
    elif 'conv' in method:
        model = Dense(model_cfg)
    elif 'CNN_GRU' in method:
        model = CNN_GRU(model_cfg)
    else:
        model = None
    return model
    

def MAE_encoder(cfg, **kwargs):
    model = PretrainVisionTransformerEncoderMine(
        img_size=(72,20),
        patch_size=(9,5),
        in_chans=cfg['channel'],
        embed_dim=cfg['hidden'],
        depth=cfg['n_layers'],
        num_heads=cfg['n_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

def RadarAE(cfg:dict, **kwargs):
    model = RadarAE_prototype(
        img_size=cfg["input_shape"][:2],
        patch_size=cfg['patch_size'],
        encoder_in_chans=cfg['channel'],
        encoder_embed_dim=cfg['hidden'],
        encoder_depth=cfg['n_layers'],
        encoder_num_heads=cfg['n_heads'],
        decoder_num_classes=cfg['patch_size'][0]*cfg['patch_size'][1]*cfg['channel'],
        decoder_embed_dim=cfg['hidden_decoder'],
        decoder_depth=cfg['n_layers_decoder'],
        decoder_num_heads=cfg['n_heads_decoder'],
        decoderType=cfg['decoderType'],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        shareWeight=cfg.get("shareWeight",False),
        isBert=cfg.get("isBert", False),
        **kwargs)
    model.default_cfg = _cfg()
    return model

def MAE(cfg, **kwargs):
    model = PretrainVisionTransformerEncoderMine(
        img_size=(72,20),
        patch_size=(9,5),
        in_chans=2,
        embed_dim=cfg['hidden'],
        depth=cfg['n_layers'],
        num_heads=cfg['n_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

