#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain.py
# @Description :
import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import Writer
import train
from models import PreTrainModelV1, buildEncoder, RadarAE
from utils import *

mode = "base"
runCfg = handle_argv_Pretrain()
# args = handle_argv('pretrain_' + mode, 'pretrain.json', mode,['v4','mmWave','base','-g','3','-s','Transformer1_linear'])
training_rate = 0.8
data, train_cfg, mask_cfg = load_pretrain_data(runCfg)

pipeline = eval(runCfg["dataset"]["pipeline"])
data_train, data_test = prepare_pretrain_dataset_mine(data, seed=train_cfg.seed)
# if runCfg['use_sup_data']:
#     data_set_train_1 = PreTrainDataset(data_train, pipeline=pipeline)
#     data_set_test = PreTrainDataset(data_test, pipeline=pipeline)

#     data_sup = np.load('dataset/mmWave/data_base.npy')
#     label_sup = np.load('dataset/mmWave/label_base.npy')
#     if isinstance(pipeline[-1], VITmask):
#         sup_pipeline = [Tshift(), Tscale(), Normalize(), Padding((72,20)), VITmask(mask_cfg)]
#     else:
#         sup_pipeline = [Tshift(), Tscale(), Normalize(), Padding((71,20)), Preprocess4Mask(mask_cfg)]
#     data_set_sup = LIBERTDataset4Pretrain(data_sup, label_sup, pipeline=sup_pipeline)
#     data_set_train = VirtualDataset(data_set_train_1,data_set_sup)
# else:
data_set_train = PreTrainDataset(data_train, pipeline=pipeline)
data_set_test = PreTrainDataset(data_test, pipeline=pipeline)
data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)


model = buildEncoder(runCfg['encoder'])
criterion = nn.MSELoss(reduction='none')

optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
device = get_device(runCfg['gpu'])
trainer = train.Trainer(train_cfg, model, optimizer, os.path.join(runCfg['save_path'],runCfg['version']), device)

def func_loss(model, batch):
    mask_seqs, masked_pos, seqs = batch #
    seq_recon = model(mask_seqs, masked_pos) #
    loss_lm = criterion(seq_recon, seqs) # for masked LM
    return loss_lm

def func_forward(model, batch):
    mask_seqs, masked_pos, seqs = batch
    seq_recon = model(mask_seqs, masked_pos)
    return seq_recon, seqs

def func_evaluate(seqs, predict_seqs):
    loss_lm = criterion(predict_seqs, seqs)
    return loss_lm.mean().cpu().numpy()

writer = Writer('Logs/Pretrain/{}'.format(runCfg['version']))
trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test
                , model_file=runCfg['pretrain_model'], writer=writer)