# -*- coding: utf-8 -*-
# @Time    : 2018/8/22 21:33
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : doc2vec.py

import os
import numpy as np
import pandas as pd
import logging

import gensim
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
# from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from utils.XCallbacks import GSCallback


class CorpusIterator(object):
    """
    语料库迭代器
    打开语料库并逐行返回数据给Word2Vec
    """
    def __init__(self, paths, column):
        """
        :param path: 文件地址
        """
        self.path = paths
        self.column = column
        dfs = []
        for p in paths:
            dfs.append(pd.read_csv(p, usecols=["id", self.column], index_col="id"))
        self.df = pd.concat(dfs)

    def __iter__(self):
        """
        迭代器
        :return: 返回每行的文章
        """
        tmp = []
        for i, line in self.df.iterrows():
            label = '%s_%s'%('TRAIN', i)
            tmp = line.tolist()[0].split(' ')
            label_text = LabeledSentence(tmp, [label])
            yield label_text


class Manager(object):

    def __init__(self, retrain=True, min_count=1, size=100, window=10, epoch=5, learning_rate=0.01):
        """
        :param result_name: 训练模型地址
        :param retrain: 是否允许多次训练
        """
        # NN params
        self.min_count = min_count  # 最小记数
        self.size = size  # 向量维度
        self.window = window  # 窗口大小
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.model = None
        # self.documents = CorpusIterator("nlp-data/train_set.csv", column="word_seg")

        self.gscallback = GSCallback()
        self.retrain = retrain

    def load_model(self, result_name):
        """
        加载模型
        :return:
        """
        try:
            self.model = Doc2Vec.load(result_name)
            logging.info('Reading pre-trained doc2vec model...')
            return self.model
        except Exception as e:
            logging.error("加载模型失败:", e)
            return

    def get_all_doc_vec(self):
        """
         获取doc2vec模型的向量
        :param model_path: 已经训练好的doc2vec模型保存路径
        :return:
        """
        model = self.model
        # vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, self.size)) for z in documents]
        # return np.concatenate(vecs)
        vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, self.size)) for z in documents]
        vecs = np.concatenate(vecs)
        doc_dict = {i:str(i) for i in range(len(vecs))}
        vecs.reshape(len(doc_dict), -1)
        return doc_dict, vecs
        # model = self.model
        # vocab = model.wv.vocab
        # word_dict = {i:word for i, word in enumerate(vocab)}
        # # TICK：加速合并np.array
        # word_vector = [model[word] for word in vocab]
        # word_vector = np.concatenate(word_vector)
        # word_vector = word_vector.reshape(len(word_dict), -1)
        # return word_dict, word_vector

    def train_model(self, documents, result_name=None):
        if not os.path.exists(result_name):
            logging.info("模型不存在, 创建新模型")
            model = Doc2Vec(documents=documents,
                            min_count=self.min_count,
                            window=self.window,
                            size=self.size,
                            sample=1e-3,
                            negative=5,
                            workers=3,
                            epochs=1,
                            compute_loss=True,
                            callbacks=[self.gscallback])
            Doc2Vec().docvecs
            # model = models.Word2Vec(sentences,
            #                         min_count=self.min_count,
            #                         size=self.size,
            #                         window=self.window,
            #                         compute_loss=True,
            #                         iter=1,
            #                         callbacks=[self.gscallback])
            self.epoch = self.epoch - 1
        elif self.retrain:
            logging.info("模型已存在, 再次训练")
            model = Doc2Vec.load(result_name)
            model.compute_loss = True
        else:
            logging.error("模型存在,禁止再次训练")
            return
        for i in range(self.epoch):
            model.train(documents,
                        total_examples=model.corpus_count,
                        start_alpha=self.learning_rate,
                        end_alpha=self.learning_rate,
                        epochs=1)

        if result_name is not None:
            logging.info("保存模型")
            self.save = model.save(result_name)

        self.model = model

    def test_model(self, function):
        """
        测试模型
        :param function 测试函数(model)传入训练模型
        :return:
        """
        if not self.model:
            logging.ERROR("模型未加载或未训练")
            return
        if not hasattr(function, "__call__"):
            logging.ERROR("未传入测试函数")
        function(self.model)

    def plot_with_labels(self, low_dim_embs, labels, filename="tsne_doc2vec.png", fonts=None):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embedding"
        plt.figure(figsize=(18, 18))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         fontproperties=fonts,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename, dpi=800)

    def plot(self):
        font = None
        try:
            font = FontProperties(fname=r"simsun.ttc", size=14)
            # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
        except Exception as e:
            logging.warning("字体不存在：", e)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 10000
        word_dict, final_embeddings = self.get_all_doc_vec()
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [word_dict[i] for i in range(plot_only)]
        self.plot_with_labels(low_dim_embs, labels, fonts=font)

    def transform2(self):
        model = self.model
        trainx = pd.read_csv('nlp-data/train_set.csv')
        # trainx = trainx.head(5)
        column = 'word_seg'
        word_dict, word_vector = manager.get_all_doc_vec()
        # maxtext = max([ len(DataHelper.intersection(row.split(" "), word_dict.values()))
        #                for i, row in trainx[column].iteritems()])
        maxtext = 4212
        xlst = []
        pad = np.zeros(shape=self.size)
        rtn = np.zeros(shape=(trainx.shape[0], maxtext, 1))
        for i, row in trainx[column].iteritems():
            words = row.split(" ")
            wordvec = [model[w] for w in words if w in word_dict.values()]
            pad_count = maxtext - len(wordvec)
            pad_lst = []
            if pad_count > 0:
                pad_lst = [pad for _ in range(pad_count)]
            wordvec += pad_lst
            xlst.append([wordvec])


        v = np.concatenate(xlst)
        return v

if __name__ == '__main__':
    from utils.commons import SUPERPARAMS
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)  # 显示INFO等级以上日志
    manager = Manager(retrain=True,
                      min_count=3,
                      size=100,
                      window=10,
                      epoch=20,
                      learning_rate=0.01)
    documents = CorpusIterator(["nlp-data/train_set.csv",
                               "nlp-data/test_set.csv"], column="word_seg")

    # manager.train_model(documents, result_name="cache/train_doc2vec.model")
    manager.load_model(result_name="cache/train_doc2vec.model")
    manager.plot()
