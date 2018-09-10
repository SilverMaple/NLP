# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 14:38
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : data_util.py

import logging
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr
from gensim.models.doc2vec import TaggedDocument
from utils.commons import SUPERPARAMS

def process_file(filename, num_classes, seq_length, field_list=["word_seg"]):
    train_set = pd.read_csv(filename, index_col=["id"])
    logging.info('Train set tag length:{}'.format(len(train_set['word_seg'])))
    # doc2vec要求文本需要加入唯一标签
    train_set.loc[:, 'word_seg'] = ['%s_%s'%('TRAIN', i) for i in range(len(train_set['word_seg']))]
    # for i in range(len(train_set['word_seg'])):
    #     label = '%s_%s' % ('TRAIN', i)
    #     # tmp = train_set['word_seg'].loc[i].split(' ')
    #     # label_text = TaggedDocument(tmp, [label])
    #     # 直接获得标签下标
    #     train_set.loc[i, 'word_seg'] = label
    logging.info('Tag documents done.')
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    n = train_set.shape[0]
    # global max_num
    # max_num = max([len(s.split(' ')) for s in train_set['word_seg']])
    # logging.info('Max num: {}'.format(max_num))
    indices = int(n*0.9)
    x_train, x_test = train_set[field_list][:indices], train_set[field_list][indices:].reset_index(drop=True)
    y_train, y_test = train_set["class"][:indices], train_set["class"][indices:]

    y_train = kr.utils.to_categorical(y_train, num_classes=num_classes)  # 将标签转换为one-hot表示
    y_test = kr.utils.to_categorical(y_test, num_classes=num_classes)  # 将标签转换为one-hot表示

    return {'train_set': (x_train, y_train), 'validate_set': (x_test, y_test)}

def read_file(filename):
    test_set = pd.read_csv(filename)
    logging.info('Read data done.')
    return test_set

def batch_iter(x, y, dv_model, batch_size=64, maxtext=39759):
    '''生成批次数据'''
    data_len = len(y)
    num_batch = int((data_len-1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    try:
        x_shuffle = x.loc[indices].reset_index(drop=True)
    except:
        pass
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1) * batch_size, data_len)
        # 左闭右闭
        x_batch = x_shuffle.loc[start_id: end_id-1].reset_index(drop=True)

        x_shuffle_vec = np.zeros(shape=(x_batch.shape[0], SUPERPARAMS.WV_SIZE, 1))
        for i in range(x_batch.shape[0]) :
            doc = x_batch.loc[i, 'word_seg']
            try:
                tmp = dv_model.docvecs[doc]
                # tmp = dv_model.infer_vector(doc)
            except Exception as e:
                logging.info(e)
                logging.info(doc)
                logging.warning('########## Document not in model. #########')
                tmp = np.zeros(shape=(100,))
            # x_shuffle_vec[i][0] = tmp
            x_shuffle_vec[i][:len(tmp)] = [[j] for j in tmp]

        yield x_shuffle_vec, y_shuffle[start_id:end_id]
