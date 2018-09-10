# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 14:38
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : data_util.py

import logging
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr
from text_cnn.cnn_model import TCNNConfig
from utils.commons import SUPERPARAMS

def process_file(filename, num_classes, seq_length, field_list=["article"]):
    # data_matrix = np.load('cache/data_matrix.npy')
    # train_set = pd.read_csv(filename)['class']
    # logging.info('Read data matrix done.')
    # train_set = train_set.sample(frac=1).reset_index(drop=True)
    # n = train_set.shape[0]
    # indices = int(n*0.9)
    # y_train, y_test = train_set[:indices], train_set[indices:]
    # x_train, x_test = data_matrix[:indices], data_matrix[indices:]

    # y_train = kr.utils.to_categorical([i-1 for i in y_train], num_classes=num_classes)  # 将标签转换为one-hot表示
    # y_test = kr.utils.to_categorical([i-1 for i in y_test], num_classes=num_classes)  # 将标签转换为one-hot表示
    train_set = pd.read_csv(filename, index_col=["id"])
    # train_set = train_set.sample(frac=0.2).reset_index(drop=True)
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    logging.info('Train_set shape:{}'.format(train_set.shape))
    n = train_set.shape[0]
    indices = int(n*0.9)
    x_train, x_test = train_set[field_list][:indices], train_set[field_list][indices:].reset_index(drop=True)
    y_train, y_test = train_set["class"][:indices], train_set["class"][indices:]

    y_train = kr.utils.to_categorical([i-1 for i in y_train], num_classes=num_classes)  # 将标签转换为one-hot表示
    y_test = kr.utils.to_categorical([i-1 for i in y_test], num_classes=num_classes)  # 将标签转换为one-hot表示

    return {'train_set': (x_train, y_train), 'validate_set': (x_test, y_test)}

def read_file(filename):
    test_set = pd.read_csv(filename)
    logging.info('Read data done.')
    return test_set

def batch_iter(x, y, wv_model, batch_size=64):
    config = TCNNConfig()
    data_len = len(y)
    num_batch = int((data_len-1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = None
    try:
        x_shuffle = x.loc[indices].reset_index(drop=True)
    except:
        pass
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1) * batch_size, data_len)
        # yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
        x_shuffle_split = [i.split(' ') for i in x_shuffle['article'][start_id:end_id]]
        for i, x in enumerate(x_shuffle_split):
            # x_shuffle_split[i] = [wv_model.wv.vocab[word].index for word in x if word in wv_model.wv.vocab]
            # plus undefine word vector
            x_shuffle_split[i] = [wv_model.wv.vocab[word].index if word in wv_model.wv.vocab else len(wv_model.wv.vocab) for word in x]
        x_shuffle_split = kr.preprocessing.sequence.pad_sequences(x_shuffle_split, config.seq_length)
        yield x_shuffle_split, y_shuffle[start_id:end_id]
