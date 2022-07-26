#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:22
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : utils.py
# @Description :

import argparse
import json
import os
import pandas as pd
from scipy.special import factorial
from torch.utils.data import Dataset
import config
from scipy.interpolate import interp1d
from config import create_io_config, load_dataset_stats, TrainConfig, MaskConfig, load_model_config
from torch.utils.tensorboard import SummaryWriter

""" Utils Functions """

import random

import numpy as np
import torch
import sys
import pickle

def load(file):
    return pickle.load(open(file,'rb'))

def save(obj, file):
    pickle.dump(obj, open(file,'wb'))

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def Writer(path):
    if os.path.exists(path):
        print("path:{} has been used!".format(path))
        exit(0)
    writer = SummaryWriter(path)
    return writer

def get_device(gpu):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def bert_mask(seq_len, goal_num_predict):
    return random.sample(range(seq_len), goal_num_predict)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)

def random_mask(numPatches, p=0.5):
    num_mask = int(np.round(p*numPatches))
    mask = np.zeros(numPatches,dtype=bool)
    mask[:num_mask] = True
    np.random.shuffle(mask)
    return mask

def merge_dataset(data, label, mode='all'):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':
            temp_label = np.unique(label[i])
            if temp_label.size == 1:
                index[i] = True
                label_new.append(label[i, 0])
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    return data[index], np.array(label_new)


def reshape_data(data, merge):
    if merge == 0:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    else:
        return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])


def reshape_label(label, merge):
    if merge == 0:
        return label.reshape(label.shape[0] * label.shape[1])
    else:
        return label.reshape(label.shape[0] * label.shape[1] // merge, merge)


def shuffle_data_label(data, label):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    return data[index, ...], label[index, ...]


# def prepare_pretrain_dataset(data, labels, training_rate, seed=None):
#     set_seeds(seed)
#     data_train, label_train, data_vali, label_vali, data_test, label_test = partition_and_reshape(data, labels, label_index=0
#                                                                                                   , training_rate=training_rate, vali_rate=0.1
#                                                                                                   , change_shape=False)
#     return data_train, label_train, data_vali, label_vali

def prepare_pretrain_dataset_mine(data, seed=None):
    set_seeds(seed)
    data_train, data_vali, data_test = partition_and_reshape_mine(data, training_rate=0.8, vali_rate=0.1)
    return data_train, data_vali

def prepare_classify_dataset_mine(data, label_rate=1.0, seed1=None, seed2=None):
    set_seeds(seed1)
    data_train, data_vali, data_test = partition_and_reshape_mine(data, training_rate=0.8, vali_rate=0.1)
    set_seeds(seed2)
    if True:
        data_train = prepare_simple_dataset_balance_mine(data_train, training_rate=label_rate)
    return data_train, data_vali, data_test

def CategorySeperate(data):
    dataA = {}
    dataB = {}
    aInds = data['labels']<3
    bInds = data['labels']>=3
    for key in data:
        if key!='size':
            dataA[key] = data[key][aInds]
            dataB[key] = data[key][bInds]
    dataA['size'] = len(dataA['labels'])
    dataB['size'] = len(dataB['labels'])
    return dataA, dataB

# def position_independent_dataset(data, label_rate=1.0, seed2=None):
#     """
#     TODO: 
#     1. position_independent_test(position position_i):
#         {
#             labeled_training_set:[samples collected in position_i in previous labeled training set],
#             validating_set:[samples collected in position_i in previous validating set],
#             testing_set:[samples not collected in position_i in previous testing set]
#         }
#     2. x-shot_test(user user_i):
#         {
#             labeled_training_set:[x samples for each label of user_i in previous labeled training set],
#             validating_set:[samples of user_i in previous validating set],
#             testing_set:[samples of user_i in previous testing set]
#         }
#     3. TBD
#     """
#     set_seeds(seed)
#     data_train, data_vali, data_test = partition_and_reshape_mine(data, training_rate=0.8, vali_rate=0.1)
#     set_seeds(seed)
#     if True:
#         data_train, data_vali, data_test = prepare_simple_dataset_balance_position(data_train, data_vali, data_test, training_rate=label_rate)
#     return data_train, data_vali, data_test

# def cross_user_dataset(data, user, label_rate=1.0, seed=None):
#     """
#     TODO: 
#     1. position_independent_test(position position_i):
#         {
#             labeled_training_set:[samples collected in position_i in previous labeled training set],
#             validating_set:[samples collected in position_i in previous validating set],
#             testing_set:[samples not collected in position_i in previous testing set]
#         }
#     2. x-shot_test(user user_i):
#         {
#             labeled_training_set:[x samples for each label of user_i in previous labeled training set],
#             validating_set:[samples of user_i in previous validating set],
#             testing_set:[samples of user_i in previous testing set]
#         }
#     3. TBD
#     """
#     set_seeds(seed)
#     data_train, data_vali, data_test = partition_and_reshape_mine(data, training_rate=0.8, vali_rate=0.1)
#     set_seeds(seed)
#     data_train, data_vali, data_test = prepare_simple_dataset_balance_crossuser(data_train, data_vali, data_test, user, training_rate=label_rate)
#     return data_train, data_vali, data_test

# def user_dataset(data, user, label_rate=1.0, seed=None):
#     """
#     TODO: 
#     1. position_independent_test(position position_i):
#         {
#             labeled_training_set:[samples collected in position_i in previous labeled training set],
#             validating_set:[samples collected in position_i in previous validating set],
#             testing_set:[samples not collected in position_i in previous testing set]
#         }
#     2. x-shot_test(user user_i):
#         {
#             labeled_training_set:[x samples for each label of user_i in previous labeled training set],
#             validating_set:[samples of user_i in previous validating set],
#             testing_set:[samples of user_i in previous testing set]
#         }
#     3. TBD
#     """
#     set_seeds(seed)
#     data_train, data_vali, data_test = partition_and_reshape_mine(data, training_rate=0.8, vali_rate=0.1)
#     set_seeds(seed)
#     data_train, data_vali, data_test = prepare_simple_dataset_balance_user(data_train, data_vali, data_test, user, training_rate=label_rate)
#     return data_train, data_vali, data_test

# def prepare_classifier_dataset(data, labels, label_index=0, training_rate=0.8, label_rate=1.0, change_shape=True
#                                , merge=0, merge_mode='all', seed=None, balance=False):

#     set_seeds(seed)
#     data_train, label_train, data_vali, label_vali, data_test, label_test \
#         = partition_and_reshape(data, labels, label_index=label_index, training_rate=training_rate, vali_rate=0.1
#                                 , change_shape=change_shape, merge=merge, merge_mode=merge_mode)
#     set_seeds(seed)
#     if balance:
#         data_train_label, label_train_label, _, _ \
#             = prepare_simple_dataset_balance(data_train, label_train, training_rate=label_rate)
#     else:
#         data_train_label, label_train_label, _, _ \
#             = prepare_simple_dataset(data_train, label_train, training_rate=label_rate)
#     return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test


# def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True
#                           , merge=0, merge_mode='all', shuffle=True):
#     arr = np.arange(data.shape[0])
#     if shuffle:
#         np.random.shuffle(arr)
#     data = data[arr]
#     labels = labels[arr]
#     train_num = int(data.shape[0] * training_rate)
#     vali_num = int(data.shape[0] * vali_rate)
#     data_train = data[:train_num, ...]
#     data_vali = data[train_num:train_num+vali_num, ...]
#     data_test = data[train_num+vali_num:, ...]
#     if labels.ndim == 2:
#         labels = labels.argmax(-1)
#         label_train = labels[:train_num]
#         label_vali = labels[train_num:train_num+vali_num]
#         label_test = labels[train_num+vali_num:]
#         return data_train, label_train, data_vali, label_vali, data_test, label_test
#     t = np.min(labels[:, :, label_index])
#     label_train = labels[:train_num, ..., label_index] - t
#     label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
#     label_test = labels[train_num+vali_num:, ..., label_index] - t
#     if change_shape:
#         data_train = reshape_data(data_train, merge)
#         data_vali = reshape_data(data_vali, merge)
#         data_test = reshape_data(data_test, merge)
#         label_train = reshape_label(label_train, merge)
#         label_vali = reshape_label(label_vali, merge)
#         label_test = reshape_label(label_test, merge)
#     if change_shape and merge != 0:
#         data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
#         data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
#         data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
#     print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
#     return data_train, label_train, data_vali, label_vali, data_test, label_test

def partition_and_reshape_mine(data, training_rate=0.8, vali_rate=0.1, shuffle=True):
    size = data['size']
    arr = np.arange(size)
    if shuffle:
        np.random.shuffle(arr)
    train_num = int(size * training_rate)
    vali_num = int(size * vali_rate)
    trainInds = arr[:train_num]
    valiInds = arr[train_num:train_num+vali_num]
    testInds = arr[train_num+vali_num:]

    data_train, data_vali, data_test = {}, {}, {}
    for key in data:
        if key!='size':
            data_train[key] = data[key][trainInds]
            data_vali[key] = data[key][valiInds]
            data_test[key] = data[key][testInds]
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (len(trainInds), len(valiInds), len(testInds)))
    return data_train, data_vali, data_test

def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    t = np.min(labels)
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    print('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
          % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
    return data_train, label_train, data_test, label_test


# def prepare_simple_dataset_balance(data, labels, training_rate=0.8):
#     labels_unique = np.unique(labels)
#     label_num = []
#     for i in range(labels_unique.size):
#         label_num.append(np.sum(labels == labels_unique[i]))
#     train_num = min(min(label_num), int(data.shape[0] * training_rate / len(label_num)))
#     if train_num == min(label_num):
#         print("Warning! You are using all of label %d." % label_num.index(train_num))
#     index = np.zeros(data.shape[0], dtype=bool)
#     for i in range(labels_unique.size):
#         class_index = np.argwhere(labels == labels_unique[i])
#         class_index = class_index.reshape(class_index.size)
#         np.random.shuffle(class_index)
#         temp = class_index[:train_num]
#         index[temp] = True
#     t = np.min(labels)
#     data_train = data[index, ...]
#     data_test = data[~index, ...]
#     label_train = labels[index, ...] - t
#     label_test = labels[~index, ...] - t
#     print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (label_train.shape[0], label_test.shape[0]
#                                                                , label_train.shape[0] * 1.0 / labels.size),flush=True)
#     return data_train, label_train, data_test, label_test

def prepare_simple_dataset_balance_mine(data, training_rate=0.8):
    labels_unique = np.unique(data['labels'])
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(data['labels'] == labels_unique[i]))
    train_num = min(min(label_num), int(data['VRSs'].shape[0] * training_rate / len(label_num)))
    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(data['labels'].shape[0], dtype=bool)
    for i in range(labels_unique.size):
        class_index = np.argwhere(data['labels'] == labels_unique[i])
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    t = np.min(data['labels'])
    data_train,data_test = {},{}
    for key in data:
        data_train[key] = data[key][index]
        data_test[key] = data[key][~index]
    print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (data_train['labels'].shape[0], data_test['labels'].shape[0]
                                                               , data_train['labels'].shape[0] * 1.0 / data_test['labels'].shape[0]),flush=True)
    return data_train

def prepare_simple_dataset_balance_position(training_data, validating_data, testing_data, training_rate=0.8):
    labels_unique = np.unique(training_data['labels'])
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(training_data['labels'] == labels_unique[i]))
    train_num = min(min(label_num), int(training_data['VRSs'].shape[0] * training_rate / len(label_num)))
    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(training_data['labels'].shape[0], dtype=bool)
    for i in range(labels_unique.size):
        indices = (training_data['labels'] == labels_unique[i]) * (training_data['positions'] == 0)
        class_index = np.argwhere(indices)
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    t = np.min(training_data['labels'])
    data_train,data_test = {},{}
    for key in training_data:
        data_train[key] = training_data[key][index]
        data_test[key] = training_data[key][~index]
    print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (data_train['labels'].shape[0], data_test['labels'].shape[0]
                                                               , data_train['labels'].shape[0] * 1.0 / data_test['labels'].shape[0]),flush=True)
    indices = validating_data['positions'] == 0
    for key in validating_data:
        validating_data[key] = validating_data[key][indices]
    indices = testing_data['positions'] != 0
    for key in testing_data:
        testing_data[key] = testing_data[key][indices]
    return data_train, validating_data, testing_data
    
def prepare_simple_dataset_balance_crossuser(training_data, validating_data, testing_data, user, training_rate=0.8):
    labels_unique = np.unique(training_data['labels'])
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(training_data['labels'] == labels_unique[i]))
    train_num = min(min(label_num), int(training_data['VRSs'].shape[0] * training_rate / len(label_num)))
    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(training_data['labels'].shape[0], dtype=bool)
    for i in range(labels_unique.size):
        indices = (training_data['labels'] == labels_unique[i]) * (training_data['users'] != user)
        class_index = np.argwhere(indices)
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    data_train,data_test = {},{}
    for key in training_data:
        data_train[key] = training_data[key][index]
        data_test[key] = training_data[key][~index]
    print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (data_train['labels'].shape[0], data_test['labels'].shape[0]
                                                               , data_train['labels'].shape[0] * 1.0 / data_test['labels'].shape[0]),flush=True)
    indices = validating_data['users'] != user
    for key in validating_data:
        validating_data[key] = validating_data[key][indices]
    indices = testing_data['users'] == user
    for key in testing_data:
        testing_data[key] = testing_data[key][indices]
    return data_train, validating_data, testing_data

def prepare_simple_dataset_balance_user(training_data, validating_data, testing_data, user, training_rate=0.8):
    labels_unique = np.unique(training_data['labels'])
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(training_data['labels'] == labels_unique[i]))
    train_num = min(min(label_num), int(training_data['VRSs'].shape[0] * training_rate / len(label_num)))
    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(training_data['labels'].shape[0], dtype=bool)
    for i in range(labels_unique.size):
        indices = (training_data['labels'] == labels_unique[i]) * (training_data['users'] == user)
        class_index = np.argwhere(indices)
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    data_train,data_test = {},{}
    for key in training_data:
        data_train[key] = training_data[key][index]
        data_test[key] = training_data[key][~index]
    print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (data_train['labels'].shape[0], data_test['labels'].shape[0]
                                                               , data_train['labels'].shape[0] * 1.0 / data_test['labels'].shape[0]),flush=True)
    indices = validating_data['users'] == user
    for key in validating_data:
        validating_data[key] = validating_data[key][indices]
    indices = testing_data['users'] == user
    for key in testing_data:
        testing_data[key] = testing_data[key][indices]
    return data_train, validating_data, testing_data

def regularization_loss(model, lambda1, lambda2):
    l1_regularization = 0.0
    l2_regularization = 0.0
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)
        l2_regularization += torch.norm(param, 2)
    return lambda1 * l1_regularization, lambda2 * l2_regularization


def match_labels(labels, labels_targets):
    index = np.zeros(labels.size, dtype=np.bool)
    for i in range(labels_targets.size):
        index = index | (labels == labels_targets[i])
    return index


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError

def PriVel(rvs, c=3e5):
    Sum = rvs.sum(1)
    nT,nV,nR = rvs.shape
    V = np.arange(-nV//2,nV//2).reshape(1,nV,1)
    rvs = rvs*V
    sumV = rvs.sum(1)
    ans = sumV/(Sum+c)
    return ans

class PriVA(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self,c1=3e5,c2=5e4):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

    def __call__(self, instance):
        VRS, ARS, label = instance
        DisStart = np.random.randint(0,3,())
        AngStart = np.random.randint(0,3,())
        VRS = VRS[...,DisStart:DisStart+20]
        ARS = ARS[:, AngStart:AngStart+8, DisStart:DisStart+20]
        priV = PriVel(VRS,c=self.c1)
        priA = PriVel(ARS,c=self.c2)
        ans = np.stack([priV,priA], -1)
        return ans, label

class PriVA_noCrop(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self,c1=3e5,c2=5e4):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

    def __call__(self, instance):
        VRS, ARS, label = instance
        priV = PriVel(VRS,c=self.c1)
        priA = PriVel(ARS,c=self.c2)
        ans = np.stack([priV,priA], -1)
        return ans, label

class PriVATest(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self,c1=3e5,c2=5e4):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

    def __call__(self, instance):
        VRS, ARS, label = instance
        DisStart = 1
        AngStart = 1
        VRS = VRS[...,DisStart:DisStart+20]
        ARS = ARS[:, AngStart:AngStart+8, DisStart:DisStart+20]
        priV = PriVel(VRS,c=self.c1)
        priA = PriVel(ARS,c=self.c2)
        ans = np.stack([priV,priA], -1)
        return ans, label

class CropTest(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self,c1=3e5,c2=5e4):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

    def __call__(self, instance):
        VRS, ARS, label = instance
        DisStart = 1
        AngStart = 1
        VRS = VRS[...,DisStart:DisStart+20]
        ARS = ARS[:, AngStart:AngStart+8, DisStart:DisStart+20]
        return VRS,ARS, label

class Tshift(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self):
        super().__init__()

    def tshift(self, input, label):
        disp = np.random.randint(-20,21,())
        T = input.shape[0]
        length = sum([(input[i]!=0).any() for i in range(T)])
        new = input.copy()
        if disp<0:
            new[length+disp:length] = 0
        elif disp>0:
            new[:-disp] = new[disp:]
            new[length-disp:] = 0
        return new, label

    def __call__(self, instance):
        instance, label = instance
        return self.tshift(instance,label)



class RTC(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self):
        super().__init__()

    def tshift(self, input, label):
        T = input.shape[0]
        length = sum([(input[i]!=0).any() for i in range(T)])
        selected_length = np.random.randint(1,length+1)
        start = np.random.randint(0,length-selected_length+1)
        new = np.zeros_like(input)
        new[:selected_length] = input[start:start+selected_length]
        return new, label

    def __call__(self, instance):
        instance, label = instance
        return self.tshift(instance,label)

class RTCRfwash(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self):
        super().__init__()

    def tshift(self, VRS,ARS):
        T = VRS.shape[0]
        length = sum([(VRS[i]!=0).any() for i in range(T)])
        selected_length = np.random.randint(1,length+1)
        start = np.random.randint(0,length-selected_length+1)
        newVRS = np.zeros_like(VRS)
        newARS = np.zeros_like(ARS)
        newVRS[:selected_length] = VRS[start:start+selected_length]
        newARS[:selected_length] = ARS[start:start+selected_length]
        return newVRS,newARS

    def __call__(self, instance):
        VRS, ARS, label = instance
        newVRS, newARS = self.tshift(VRS,ARS)
        return newVRS, newARS, label




class Padding(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, instance):
        new, label = instance
        new = new.copy()
        H0,W0,C = new.shape
        H, W = self.size
        new = np.concatenate([new, np.zeros((H-H0,W0,C))], 0)
        if W0 > W:
            new = new[:,4:]
        return new, label

class PaddingRVS(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, nT):
        super().__init__()
        self.nT = nT

    def __call__(self, instance):
        VRS, _, label = instance
        DisStart = 1
        VRS = VRS[...,DisStart:DisStart+20]
        length = VRS.shape[0]
        VRS = np.concatenate([VRS,np.zeros([self.nT-length, *VRS.shape[1:]])],0)
        return VRS, label

class Normalize(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        new, label = instance
        T = new.shape[0]
        length = sum([(new[i]!=0).any() for i in range(T)])

        instance_new = new.copy()
        avg = new[:length,:,0].mean()
        std = new[:length,:,0].std()
        instance_new[:length,:,0] = (new[:length,:,0]-avg)/std
        avg = new[:length,:,1].mean()
        std = new[:length,:,1].std()
        instance_new[:length,:,1] = (new[:length,:,1]-avg)/std
        return instance_new, label

class Flatten(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self):
        super().__init__()

    def __call__(self, instance:torch.Tensor):
        T,R,C = instance.shape
        instance = instance.swapaxes(2,1)
        instance_new = instance.reshape(T,R*C)
        return instance_new

class getRV(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self):
        super().__init__()

    def __call__(self, instance:torch.Tensor):
        sample, label = instance
        return sample[...,0:1], label

class getRA(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self):
        super().__init__()

    def __call__(self, instance:torch.Tensor):
        sample, label = instance
        return sample[...,1:], label

class Preprocess4Normalization(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        instance_new = instance.copy()[:, :self.feature_len]
        if instance_new.shape[1] >= 6 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new


class Preprocess4Mask:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        instance, label = instance
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        seqLen = instance.shape[0]
        # seqLen = np.sum((instance[i]!=0).any() for i in range(shape[0]))
        mask_pos = span_mask(seqLen, self.max_gram,  goal_num_predict=n_pred)

        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), *shape[1:]))
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), *shape[1:]))
        seq = instance[mask_pos_index, :]
        return instance_mask, np.array(mask_pos_index), np.array(seq)

class VITmask:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.patch_size = mask_cfg.patch_size
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        instance, label = instance
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        H, W, C = instance.shape
        H_, W_ = self.patch_size
        P1 = H//H_
        P2 = W//W_
        seq = instance.reshape(P1, H_, P2, W_, C)
        seq = seq.transpose(0, 2, 1, 3, 4).reshape(P1*P2, H_*W_*C) # P1 x P2 x H_ x W_ x 2
        seqLen = seq.shape[0]
        # seqLen = np.sum((instance[i]!=0).any() for i in range(shape[0]))
        mask = random_mask(seqLen,p=self.mask_ratio)

        seq_masked = seq.copy()
        if np.random.rand() < self.mask_prob:
            seq_masked[mask, :] = 0
        elif np.random.rand() < self.replace_prob:
            seq_masked[mask, :] = np.random.random((len(mask), *shape[1:]))
        seq = seq[mask]
        return seq_masked, np.array(mask,dtype=bool), np.array(seq)

class BERTmask:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        instance, label = instance
        shape = instance.shape

        H, W, C = instance.shape
        seq = instance.reshape(H, W*C)
        seqLen = H
        mask = random_mask(seqLen,p=self.mask_ratio)

        seq_masked = seq.copy()
        if np.random.rand() < self.mask_prob:
            seq_masked[mask, :] = 0
        elif np.random.rand() < self.replace_prob+self.mask_prob:
            seq_masked[mask, :] = np.random.random((mask.sum(), np.prod(shape[1:])))
        seq = seq[mask]
        return seq_masked, np.array(mask,dtype=bool), np.array(seq)

class VITInput:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_cfg):
        if isinstance(mask_cfg, list):
            self.patch_size = mask_cfg
        else:
            self.patch_size = mask_cfg.patch_size

    def __call__(self, instance):
        instance, label = instance
        shape = instance.shape

        H, W, C = instance.shape
        H_, W_ = self.patch_size
        P1 = H//H_
        P2 = W//W_
        seq = instance.reshape(P1, H_, P2, W_, C)
        seq = seq.transpose(0, 2, 1, 3, 4).reshape(P1*P2, H_*W_*C) # P1 x P2 x H_ x W_ x 2
        seqLen = seq.shape[0]
        # seqLen = np.sum((instance[i]!=0).any() for i in range(shape[0]))
        return seq, label

class SoliFeature(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        VRS,  label = instance
        Vcen = Vcentroid(VRS)
        dis = disp(Vcen)
        vdis = Vdis(VRS,Vcen)
        rdis = Rdis(VRS)
        rp = RP(VRS)
        eins = Eins(rp)
        etv = Etv(rp)
        ans = np.concatenate([Vcen,dis,vdis,rdis,eins,etv], 0) # size:[6*nT]
        return ans, label

def Vcentroid(VRS):
    nT, nV, nR = VRS.shape
    mat:np.ndarray = np.arange(0,nV).reshape(1,nV,1)-nV/2
    VRS_ = mat*VRS
    ans = VRS.sum((1,2))/(VRS_.sum((1,2))+1e-5)
    return ans 

def disp(Vcen:np.ndarray):
    nT = Vcen.shape[0]
    ans = np.zeros(nT)
    ans[0] = Vcen[0]
    for i in range(1,nT):
        ans[i] = ans[i-1]+Vcen[i]
    return ans

def Vdis(VRS:np.ndarray, Vcen:np.ndarray):
    nT, nV, nR = VRS.shape
    v = np.arange(0,nV).reshape(1,nV,1)-nV/2
    Vcen = Vcen.reshape(nT,1,1)
    dis = (v-Vcen)**2
    mDis = ((dis*VRS).sum((1,2))/(VRS.sum((1,2))+1e-5))**0.5
    return mDis

def RP(VRS:np.ndarray):
    return VRS.mean(1)

def Ri(rp:np.ndarray):
    return rp.argmax(1)

def Rdis(VRS:np.ndarray):
    nT, nV, nR = VRS.shape
    rp = RP(VRS)
    r = np.arange(0,nR).reshape(1,nR)
    ri = Ri(rp).reshape(nT,1)
    dis = (r-ri)**2
    mDis = ((rp*dis).sum(1)/(rp.sum(1)+1e-5))**0.5
    return mDis

def Eins(rp:np.ndarray):
    return rp.mean(1)

def Etv(rp:np.ndarray):
    nT, nR = rp.shape
    ans = rp[1:]-rp[:-1]
    ans = np.abs(ans).mean(1)
    ans = np.concatenate([ans,np.zeros([1])],0)
    return ans

class IMUDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance).float(), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)

class DatasetMine(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = [self.data['VRSs'][index], self.data['ARSs'][index], self.data['labels'][index]] #self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance[0]).float(), torch.from_numpy(np.array(instance[1])).long()

    def __len__(self):
        return len(self.data['VRSs'])

class ARSVRSDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = [self.data['VRSs'][index], self.data['ARSs'][index], self.data['labels'][index]] #self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance[0]).float(), torch.from_numpy(instance[1]).float(), torch.from_numpy(np.array(instance[2])).long()

    def __len__(self):
        return len(self.data['VRSs'])

class FFTDataset(Dataset):
    def __init__(self, data, labels, mode=0, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels
        self.mode = mode

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        seq = self.preprocess(instance)
        return torch.from_numpy(seq), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)

    def preprocess(self, instance):
        f = np.fft.fft(instance, axis=0, n=10)
        mag = np.abs(f)
        phase = np.angle(f)
        return np.concatenate([mag, phase], axis=0).astype(np.float32)

class LIBERTDataset4Pretrain(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, label, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.label = label

    def __getitem__(self, index):
        instance = self.data[index],self.label[index]
        if sum(self.data[index].all((1,2)))<20:
            import pdb;pdb.set_trace()
        for proc in self.pipeline:
            instance = proc(instance)
        mask_seq, masked_pos, seq = instance
        if isinstance(self.pipeline[-1], Preprocess4Mask):
            return torch.from_numpy(mask_seq).float(), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq).float()
        else:
            return torch.from_numpy(mask_seq).float(), torch.from_numpy(masked_pos).bool(), torch.from_numpy(seq).float()


    def __len__(self):
        return len(self.data)

class PreTrainDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data 

    def __getitem__(self, index):
        instance = [self.data['VRSs'][index], self.data['ARSs'][index], self.data['labels'][index]] #self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        mask_seq, masked_pos, seq = instance
        if isinstance(self.pipeline[-1], Preprocess4Mask):
            return torch.from_numpy(mask_seq).float(), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq).float()
        else:
            return torch.from_numpy(mask_seq).float(), torch.from_numpy(masked_pos).bool(), torch.from_numpy(seq).float()

    def __len__(self):
        return len(self.data['VRSs'])

class VirtualDataset(Dataset):
    def __init__(self, dataset1, dataset2) -> None:
        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)

    def __getitem__(self, index):
        if index<self.len1:
            return self.dataset1[index]
        else:
            return self.dataset2[index-self.len1]
    
    def __len__(self):
        return self.len1+self.len2

def handle_argv(target, config_train, prefix, manual=None):
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('model_version', type=str, help='Model config')
    parser.add_argument('dataset', type=str, help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib','mmWave'])
    parser.add_argument('dataset_version',  type=str, help='Dataset version', choices=['10_100', '20_120','base'])
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    parser.add_argument('-f', '--model_file', type=str, default=None, help='Pretrain model file')
    parser.add_argument('-t', '--train_cfg', type=str, default='./config/' + config_train, help='Training config json file path')
    parser.add_argument('-a', '--mask_cfg', type=str, default='./config/mask.json',
                        help='Mask strategy json file path')
    parser.add_argument('-l', '--label_index', type=int, default=-1,
                        help='Label Index')
    parser.add_argument('-s', '--save_model', type=str, default='model',
                        help='The saved model name')
    # try:
    if manual:
        args = parser.parse_args(manual)
    else:
        args = parser.parse_args()
    model_cfg = load_model_config(target, prefix, args.model_version)
    if model_cfg is None:
        print("Unable to find corresponding model config!")
        sys.exit()
    args.model_cfg = model_cfg
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg
    args = create_io_config(args, args.dataset, args.dataset_version, pretrain_model=args.model_file, target=target)
    return args
    # except:
    #     parser.print_help()
    #     sys.exit(0)

def handle_argv_Pretrain(manual=None):
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('version', type=str, help='run config')
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    args = parser.parse_args(manual)
    run_path = 'config/runPretrain.json'
    runJson = json.load(open(run_path, "r"))
    ans = runJson[args.version]
    ans['gpu'] = args.gpu
    ans['version'] = args.version
    return ans

def handle_argv_Classify(manual=None):
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('version', type=str, help='run config')
    parser.add_argument('labelRate', type=str, help='label rate')
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    args = parser.parse_args(manual)
    run_path = 'config/runClassify.json'
    pretrain_path = 'config/runPretrain.json'
    runJson = json.load(open(run_path, "r"))
    pretrainJson = json.load(open(pretrain_path, "r"))
    if '@' in args.version:
        pos = args.version.find('@')
        step = args.version[pos+1:]
        args.version = args.version[:pos]
    else:
        step = None
    ans = runJson[args.version]
    ans['pretrainCfg'] = pretrainJson[ans["pretrainCfgVersion"]]
    ans['gpu'] = args.gpu
    ans['version'] = args.version
    ans['step'] = step
    ans['label_rate'] = float(args.labelRate)
    if ans['gpu'] == None:
        ans['gpu'] = str(np.random.randint(0,4,()))
    return ans

def handle_argv_benchmark(manual=None):
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('version', type=str, help='run config')
    parser.add_argument('labelRate', type=str, help='label rate')
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    args = parser.parse_args(manual)
    run_path = 'config/runBenchmark.json'
    runJson = json.load(open(run_path, "r"))
    if '@' in args.version:
        pos = args.version.find('@')
        step = args.version[pos+1:]
        args.version = args.version[:pos]
    else:
        step = None
    ans = runJson[args.version]
    ans['gpu'] = args.gpu
    ans['version'] = args.version
    ans['step'] = step
    ans['label_rate'] = float(args.labelRate)
    if ans['gpu'] == None:
        ans['gpu'] = str(np.random.randint(0,4,()))
    return ans


def load_pretrain_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    dataset_cfg = args.dataset_cfg
    if model_cfg.feature_num > dataset_cfg.dimension:
        print("Bad Crossnum in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg

def loadData(path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data/data")
    return load(data_path)


def load_pretrain_data(args):
    dataset = args['dataset']
    path = dataset['path']
    pretrain_cfg = TrainConfig.from_dict(args['pretrainCfg'])
    mask_cfg = MaskConfig.from_dict(args['maskCfg'])
    set_seeds(pretrain_cfg.seed)
    data = loadData(path)

    return data, pretrain_cfg, mask_cfg

def load_classify_data(runCfg):
    PretrainCfg = runCfg['pretrainCfg']
    dataset = PretrainCfg['dataset']
    path = dataset['path']
    pretrain_cfg = TrainConfig.from_dict(PretrainCfg['pretrainCfg'])
    classify_cfg = TrainConfig.from_dict(runCfg['classifyCfg'])
    set_seeds(pretrain_cfg.seed)
    data = loadData(path)

    return data, pretrain_cfg, classify_cfg

def load_benchmark_data(runCfg):
    benchmark_cfg = TrainConfig.from_dict(runCfg['classifyCfg'])
    set_seeds(benchmark_cfg.seed)
    data = loadData('packs/')

    return data, benchmark_cfg

def load_classifier_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, dataset_cfg


def load_classifier_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg, dataset_cfg


def load_bert_classifier_data_config(args):
    model_bert_cfg, model_classifier_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    if model_bert_cfg.feature_num > dataset_cfg.dimension:
        print("Bad feature_num in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
