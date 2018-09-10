# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 14:38
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : data_util.py

import logging
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr
from sklearn.manifold.t_sne import TSNE
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from text_cnn.cnn_model import TCNNConfig
from utils.commons import SUPERPARAMS



def process_file(filename, num_classes, seq_length, field_list=["word_seg"]):
    train_set = pd.read_csv(filename, index_col=["id"])
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    n = train_set.shape[0]
    global max_num
    max_num = max([len(s.split(' ')) for s in train_set['word_seg']])
    logging.info('Max num: {}'.format(max_num))
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

def batch_iter(x, y, batch_size=64, maxtext=39759):
    '''生成批次数据'''
    config = TCNNConfig()
    data_len = len(y)
    num_batch = int((data_len-1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    # 词向量过大，不能一次性读取，需分批使用到时再进行读取
    x_shuffle = None
    try:
        x_shuffle = x.loc[indices].reset_index(drop=True)
    except:
        pass
    y_shuffle = y[indices]

    # 获取tf_idf特征矩阵
    logging.info('generating Tf-Idf matrix...')
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    train_set_term_doc = vec.fit_transform(x_shuffle['word_seg'])
    logging.info('generating done.{}'.format(train_set_term_doc.shape))
    # logging.info('generating done. {}'.format(train_set_term_doc.shape))
    # pca降维
    # tsne = TSNE(perplexity=30, n_components=config.seq_length, method='exact', init='pca', n_iter=5000)
    # train_set_term_doc = tsne.fit_transform(train_set_term_doc)
    # logging.info('PCA:{}'.format(train_set_term_doc.shape))

    # 截断SVD，截断奇异值分解，是另一种降维方法，其可以用在稀疏矩阵而PCA不能
    svd = TruncatedSVD(config.seq_length, algorithm='arpack')

    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1) * batch_size, data_len)
        # x_batch = [ [i] for i in train_set_term_doc[start_id : end_id]]
        x_batch = train_set_term_doc[start_id : end_id]
        logging.info('x_batch extract:{}, {}~{}'.format(x_batch.shape, start_id, end_id))
        # x_batch = kr.preprocessing.sequence.pad_sequences(x_batch, config.vocab_size)
        x_batch = svd.fit_transform(x_batch)
        x_batch = kr.preprocessing.sequence.pad_sequences(x_batch, config.seq_length)
        logging.info('TruncatedSVD fit_transform done.{}'.format(x_batch.shape))
        x_vec = np.empty(shape=(end_id-start_id, config.seq_length, 1))
        for j in range(len(x_batch)):
            x_vec[j] = np.array([[i] for i in x_batch[j]])
        # if end_id == data_len:
        #     x_vec = None
        # else:
        #     # x_batch = tsne.fit_transform(x_batch)
        #     x_batch = svd.fit_transform(x_batch)
        #     logging.info('TruncatedSVD fit_transform done.')
        #     x_vec = np.empty(shape=(config.batch_size, config.seq_length, 1))
        #     for j in range(len(x_batch)):
        #         x_vec[j] = np.array([[i] for i in x_batch[j]])
        '''拟合数据'''
        # K = config.seq_length  # 要降的维度
        # pca_model = pca.PCA(n_components=K).fit(x_batch)  # 拟合数据，n_components定义要降的维度
        # x_batch = pca_model.transform(x_batch)  # transform就会执行降维操作
        # x_batch = pca.PCA(n_components=K).fit_transform(x_batch)
        yield x_vec, y_shuffle[start_id : end_id]
        # x_batch = x_shuffle.loc[start_id: end_id-1].reset_index(drop=True)
        # x_batch = x_batch[x_batch.columns[0]].str.split(" ")
        #
        # x_shuffle_vec = np.zeros(shape=(x_batch.shape[0], SUPERPARAMS.WV_SIZE, 1))
        # for i in range(x_batch.shape[0]) :
        #     words = x_batch.loc[i]
        #     tmp = [wv_model[w] for w in words if w in wv_model]
        #     tmp = (np.sum(tmp, axis=0) / len(tmp)).transpose()
        #     x_shuffle_vec[i][:len(tmp)] = [[j] for j in tmp]
        #
        # yield x_shuffle_vec, y_shuffle[start_id:end_id]

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