# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 14:38
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : data_util.py

import numpy as np
import logging
from text_cnn.cnn_model import TCNNConfig
import pandas as pd
import tensorflow.contrib.keras as kr

def process_file(filename, num_classes, seq_length, field_list=["word_seg"]):
    train_set = pd.read_csv(filename, index_col=["id"])
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    n = train_set.shape[0]
    indices = int(n * 0.9)
    x_train, x_test = train_set[field_list][:indices], train_set[field_list][indices:].reset_index(drop=True)
    y_train, y_test = train_set["class"][:indices], train_set["class"][indices:]
    y_train = kr.utils.to_categorical([i-1 for i in y_train], num_classes=num_classes)  # 将标签转换为one-hot表示
    y_test = kr.utils.to_categorical([i-1 for i in y_test], num_classes=num_classes)  # 将标签转换为one-hot表示
    return {'train_set': (x_train, y_train), 'validate_set': (x_test, y_test)}

def read_file(filename):
    test_set = pd.read_csv(filename)
    n = test_set.shape[0]
    test_set_data = []
    test_set_id = []
    for i in range(n):
        test_set_data.append(test_set['word_seg'][i].split(' '))
        test_set_id.append(test_set['id'][i])
    return {'data': test_set_data, 'id': test_set_id}

def batch_iter(x, y, batch_size=64):
    config = TCNNConfig()
    seq_length = config.seq_length
    data_len = len(x)
    num_batch = int((data_len-1) / batch_size) + 1

    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1) * batch_size, data_len)
        x_shuffle = x['word_seg'][start_id:end_id]
        x_shuffle = [str(i).split(' ') for i in x_shuffle]
        x_shuffle_pad = kr.preprocessing.sequence.pad_sequences(x_shuffle, maxlen=seq_length)
        yield x_shuffle_pad, y[start_id:end_id]