# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 14:38
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : data_util.py

import logging
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr

from utils.commons import SUPERPARAMS



def process_file(filename, num_classes, seq_length, field_list=["word_seg"]):
    train_set = pd.read_csv(filename, index_col=["id"])
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    n = train_set.shape[0]
    global max_num
    max_num = max([len(s.split(' ')) for s in train_set['word_seg']])
    logging.info('Max num: {}'.format(max_num))
    # x_train, y_train, x_val, y_val = [], [], [], []
    # for i in range(n):
    #     # 分词
    #     words = train_set['word_seg'][i].split(' ')
    #     # 10:1比例划分训练集与验证集
    #     if i % 10 == 0:
    #         y_val.append(train_set['class'][i])
    #         x_val.append(words)
    #     else:
    #         y_train.append(train_set['class'][i])
    #         x_train.append(words)
    indices = int(n*0.9)
    x_train, x_test = train_set[field_list][:indices], train_set[field_list][indices:].reset_index(drop=True)
    y_train, y_test = train_set["class"][:indices], train_set["class"][indices:]

    y_train = kr.utils.to_categorical(y_train, num_classes=num_classes)  # 将标签转换为one-hot表示
    y_test = kr.utils.to_categorical(y_test, num_classes=num_classes)  # 将标签转换为one-hot表示

    return {'train_set': (x_train, y_train), 'validate_set': (x_test, y_test)}

def process_file_deprecated(filename, num_classes, seq_length):
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
    logging.info('Read data done.')
    return test_set
    # n = test_set.shape[0]
    # test_set_data = []
    # test_set_id = []
    # for i in range(n):
    #     test_set_data.append(test_set['word_seg'][i].split(' '))
    #     test_set_id.append(test_set['id'][i])
    #
    # return {'data': test_set_data, 'id': test_set_id}

def batch_iter(x, y, wv_model, batch_size=64, maxtext=39759):
    '''

    :param x: ds
    :param y: 类别数组
    :param batch_size:
    :return:
    '''
    '''生成批次数据'''
    data_len = len(y)
    num_batch = int((data_len-1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    # 词向量过大，不能一次性读取，需分批使用到时再进行读取
    # x_shuffle = x[indices]
    try:
        x_shuffle = x.loc[indices].reset_index(drop=True)
    except:
        pass
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1) * batch_size, data_len)
        x_batch = x_shuffle.loc[start_id: end_id-1].reset_index(drop=True)
        x_batch = x_batch[x_batch.columns[0]].str.split(" ")

        # 取出x_batch对应的词向量
        # x_shuffle_vec = np.zeros(shape=(x_batch.shape[0], maxtext, SUPERPARAMS.WV_SIZE))
        # for i in range(x_batch.shape[0]) :
        #     words = x_batch.loc[i]
        #     tmp = [wv_model[w] for w in words if w in wv_model]
        #     x_shuffle_vec[i][:len(tmp)] = tmp

        x_shuffle_vec = np.zeros(shape=(x_batch.shape[0], SUPERPARAMS.WV_SIZE, 1))
        for i in range(x_batch.shape[0]) :
            words = x_batch.loc[i]
            tmp = [wv_model[w] for w in words if w in wv_model]
            tmp = (np.sum(tmp, axis=0) / len(tmp)).transpose()
            x_shuffle_vec[i][:len(tmp)] = [[j] for j in tmp]

        yield x_shuffle_vec, y_shuffle[start_id:end_id]

def batch_iter_deprecated(x, y, batch_size=64):
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