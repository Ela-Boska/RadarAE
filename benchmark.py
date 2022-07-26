# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/1/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : embedding.py
# @Description : check the output and embedding of pretrained model
import os
import sys
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import Writer, handle_argv_benchmark, load_benchmark_data
import train
from config import load_dataset_label_names
from models import buildEncoder, buildDecoder, CompositeModel
# from plot import plot_reconstruct_sensor, plot_embedding
from utils import *
from statistic import stat_acc_f1, stat_results

training_rate = 0.8

def fetch_setup(runCfg,step):
    data, benchmark_cfg = load_benchmark_data(runCfg)
    pipeline_train = eval(runCfg["dataset"]["pipelineTrain"])
    pipeline_vali = eval(runCfg["dataset"]["pipelineVali"])
    pipeline_test = eval(runCfg["dataset"]["pipelineTest"])
    # if runCfg['pi']:
    #     data_train, data_vali, data_test = position_independent_dataset(data, label_rate=label_rate, seed1=benchmark_cfg.seed, seed2=benchmark_cfg.seed+step*23)
    # elif runCfg['crossuser']:
    #     print("cross user test, target user {}".format(runCfg['targetUser']))
    #     data_train, data_vali, data_test = cross_user_dataset(data, runCfg['targetUser'], label_rate=label_rate, seed=benchmark_cfg.seed+step*23)
    # elif runCfg['user']:
    #     print("user test, target user {}".format(runCfg['targetUser']))
    #     data_train, data_vali, data_test = user_dataset(data, runCfg['targetUser'], label_rate=label_rate, seed=benchmark_cfg.seed+step*23)
    # else:
    data_train, data_vali, data_test = prepare_classify_dataset_mine(data, label_rate=label_rate, seed1=benchmark_cfg.seed, seed2=benchmark_cfg.seed+step*23)
    if 'origin' in runCfg['dataset']:
        data_set_train = ARSVRSDataset(data_train, pipeline=pipeline_train)
        data_set_vali = ARSVRSDataset(data_vali, pipeline=pipeline_vali)
        data_set_test = ARSVRSDataset(data_test, pipeline=pipeline_test)
    else:
        data_set_train = DatasetMine(data_train, pipeline=pipeline_train)
        data_set_vali = DatasetMine(data_vali, pipeline=pipeline_vali)
        data_set_test = DatasetMine(data_test, pipeline=pipeline_test)
    data_loader_train = DataLoader(data_set_train, shuffle=False, batch_size=benchmark_cfg.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=benchmark_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=benchmark_cfg.batch_size)
    model = buildDecoder(runCfg['decoder'])
    return data_loader_train, data_loader_vali, data_loader_test, model, benchmark_cfg


def main(runCfg,step):
    data_loader_train, data_loader_vali, data_loader_test, model, benchmark_cfg = fetch_setup(runCfg,step)

    optimizer = optim.Adam(params=model.parameters(), lr=benchmark_cfg.lr)  # , weight_decay=0.95
    trainer = train.Trainer(benchmark_cfg, model, optimizer, None, get_device(runCfg['gpu']))
    if "loss" in runCfg:
        criterion = nn.MultiMarginLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    def func_loss(model:nn.Module, batch):
        model.train()
        if len(batch)==2:
            inputs, label = batch
            logits = model(inputs)
        elif len(batch)==3:
            inputs1, inputs2, label = batch
            logits = model(inputs1,inputs2)
        loss = criterion(logits, label)
        model.eval()
        return loss

    def func_forward(model:nn.Module, batch):
        model.eval()
        if len(batch)==2:
            inputs, label = batch
            logits = model(inputs)
        elif len(batch)==3:
            inputs1, inputs2, label = batch
            logits = model(inputs1,inputs2)
        return logits, label

    def func_evaluate(label, predicts):
        stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
        return stat

    writer = None #Writer('Logs/Classify/{}'.format(os.path.join(runCfg['version'],str(label_rate))))
    _, _, test_acc, _, _, test_f1 = trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali,writer=writer)
    label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    return test_acc, test_f1

save = True
mode = "base"
runCfg = handle_argv_benchmark()
label_rate = runCfg['label_rate']
if runCfg['pi']:
    fileName = "log/benchmark/{}_{}_PI.log".format(runCfg['version'],label_rate)
elif runCfg['crossuser']:
    fileName = "log/benchmark/{}_{}_CU.log".format(runCfg['version'],label_rate)
elif runCfg['user']:
    fileName = "log/benchmark/{}_{}_U.log".format(runCfg['version'],label_rate)
else:
    fileName = "log/benchmark/{}_{}.log".format(runCfg['version'],label_rate)
if os.path.exists(fileName):
    print("{} already exists!".format(fileName))
    exit(0)
sys.stdout = open(fileName,'w')
if runCfg['crossuser'] or runCfg['user']:
    for i in range(4):
        runCfg['targetUser'] = i
        main(runCfg)
else:
    acc, f1 = [],[]
    for i in range(5):
        test_acc, test_f1 = main(runCfg,i)
        acc.append(test_acc)
        f1.append(test_f1)
    acc = np.array(acc)
    f1 = np.array(f1)
    print("acc={},f1={}".format(acc,f1), flush=True)
    print('acc: {} +- {}, f1: {} +- {}'.format(acc.mean(),acc.std(),f1.mean(),f1.std()), flush=True)