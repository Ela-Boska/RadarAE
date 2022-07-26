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
from utils import Writer
import train
from config import load_dataset_label_names,MaskConfig
from models import buildEncoder, buildDecoder, CompositeModel
# from plot import plot_reconstruct_sensor, plot_embedding
from utils import *
from statistic import stat_acc_f1, stat_results

def fetch_setup(runCfg,step):
    data, pretrain_cfg, classify_cfg = load_classify_data(runCfg)
    mask_cfg = MaskConfig.from_dict(runCfg['pretrainCfg']['maskCfg'])
    
    pipeline_train = eval(runCfg["dataset"]["pipelineTrain"])
    pipeline_vali = eval(runCfg["dataset"]["pipelineVali"])
    pipeline_test = eval(runCfg["dataset"]["pipelineTest"])
    data_train, data_vali, data_test = prepare_classify_dataset_mine(data, label_rate=label_rate, seed1=pretrain_cfg.seed, seed2=pretrain_cfg.seed+step*23)
    data_set_train = DatasetMine(data_train, pipeline=pipeline_train)
    data_set_vali = DatasetMine(data_vali, pipeline=pipeline_vali)
    data_set_test = DatasetMine(data_test, pipeline=pipeline_test)
    data_loader_train = DataLoader(data_set_train, shuffle=False, batch_size=classify_cfg.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=classify_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=classify_cfg.batch_size)
    if runCfg['step'] is not None:
        weight_file = os.path.join(runCfg['pretrainCfg']['save_path'],runCfg["pretrainCfgVersion"]+"_"+runCfg['step'])
    else:
        weight_file = os.path.join(runCfg['pretrainCfg']['save_path'],runCfg["pretrainCfgVersion"])
    encoder = buildEncoder(runCfg['pretrainCfg']['encoder'],output_embed=True,model_file=weight_file)
    decoder = buildDecoder(runCfg['decoder'])
    model = CompositeModel(encoder, decoder)
    return data_loader_train, data_loader_vali, data_loader_test, model, pretrain_cfg, classify_cfg


def main(runCfg,step):
    data_loader_train, data_loader_vali, data_loader_test, model, pretrain_cfg, classify_cfg \
        = fetch_setup(runCfg,step)

    optimizer = optim.Adam(params=model.parameters(), lr=classify_cfg.lr)  # , weight_decay=0.95
    trainer = train.Trainer(classify_cfg, model, optimizer, "saved/Classify/{}".format(runCfg['version']), get_device(runCfg['gpu']))
    criterion = nn.CrossEntropyLoss()

    def func_loss(model:nn.Module, batch):
        inputs, label = batch
        model.train()
        logits = model(inputs)
        loss = criterion(logits, label)
        model.eval()
        return loss

    def func_forward(model:nn.Module, batch):
        model.eval()
        inputs, label = batch
        logits = model(inputs)
        return logits, label

    def func_evaluate(label, predicts):
        stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
        return stat

    writer = None #Writer('Logs/Classify/{}'.format(os.path.join(runCfg['version'],str(label_rate))))
    _, _, test_acc, _, _, test_f1 = trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali,writer=writer)
    # label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    return test_acc, test_f1

training_rate = 0.8
save = True
mode = "base"
runCfg = handle_argv_Classify()
label_rate = runCfg['label_rate']
fileName = f"log/classify/{runCfg['version']}_{label_rate}.log"
if os.path.exists(fileName):
    print("{} already exists!".format(fileName))
    exit(0)
sys.stdout = open(fileName,'w')


acc, f1 = [],[]
for i in range(5):
    test_acc, test_f1 = main(runCfg,i)
    acc.append(test_acc)
    f1.append(test_f1)
acc = np.array(acc)
f1 = np.array(f1)
print(f"acc={acc},f1={f1}", flush=True)
print(f'acc: {acc.mean()} +- {acc.std()}, f1: {f1.mean()} +- {f1.std()}', flush=True)