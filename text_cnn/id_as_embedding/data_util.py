# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 14:38
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : data_util.py

import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr

def process_file(filename, num_classes, seq_length):
    train_set = pd.read_csv(filename)
    n = train_set.shape[0]
    train_set_data = {'train_set': [], 'validate_set': []}
    for i in range(n):
        # 分词
        words = train_set['word_seg'][i].split(' ')
        # 10:1比例划分训练集与验证集
        if i % 10 == 0:
            train_set_data['validate_set'].append((words, train_set['class'][i]))
        else:
            train_set_data['train_set'].append((words, train_set['class'][i]))

    # 随机打乱数据顺序, 可以在处理数据时再打乱
    # random.shuffle(train_set_data['train_set'])
    # random.shuffle(train_set_data['validate_set'])

    x_train, y_train, x_val, y_val = [], [], [], []
    x_train[:], y_train[:] = zip(*train_set_data['train_set'])
    x_val[:], y_val[:] = zip(*train_set_data['validate_set'])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_train = kr.preprocessing.sequence.pad_sequences(x_train, seq_length)
    y_train = kr.utils.to_categorical(y_train, num_classes=num_classes)  # 将标签转换为one-hot表示
    x_val = kr.preprocessing.sequence.pad_sequences(x_val, seq_length)
    y_val = kr.utils.to_categorical(y_val, num_classes=num_classes)  # 将标签转换为one-hot表示

    return {'train_set': (x_train, y_train), 'validate_set': (x_val, y_val)}

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
    '''生成批次数据'''
    data_len = len(x)
    num_batch = int((data_len-1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]