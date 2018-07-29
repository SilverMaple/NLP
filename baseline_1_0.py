# -*- coding: utf-8 -*-
# @Time    : 2018/7/28 15:07
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : baseline_1_0.py

import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
# import logging
#
#
# # 获取日志信息
# logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.INFO)

print('Start...')
t0 = time.time()
t1 = time.time()
train_set = pd.read_csv('input/train_set.csv') #id,article,word_seg,class
print('train_set.csv read done: ', time.time()-t1)
t1 = time.time()
test_set = pd.read_csv('input/test_set.csv') #id,article,word_seg
print('test_set.csv read done: ', time.time()-t1)
for i in test_set:
    print(i)
t1 = time.time()
test_id = test_set['id'].copy()


column = 'word_seg'
n = train_set.shape[0] # shape = [n, 4], 即获取数据行数
# 转为TF-IDF文字特征矩阵，ngram取值为（1，2），max_df和min_df取值中0.0~1.0表示百分比，整型表示绝对数量
# use_idf=1表示启用逆文档频率重新加权，smooth_idf表示权重加一避免除零，sublinear_tf为替换tf为1 + log(tf)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
train_set_term_doc = vec.fit_transform(train_set[column])
test_set_term_doc = vec.transform(test_set[column])

y = (train_set['class']-1).astype(int)
clf = LogisticRegression(C=4, dual=True)
clf.fit(train_set_term_doc, y) # 训练模型与类型对应
print('train done: ', time.time()-t1)
t1 = time.time()
preds = clf.predict_proba(test_set_term_doc)
print('prediction done: ', time.time()-t1)
t1 = time.time()

# 保存概率文件
test_prob = pd.DataFrame(preds)
test_prob.columns = ['class_prob_%s'%i for i in range(1, preds.shape[1]+1)]
test_prob['id'] = list(test_id)
test_prob.to_csv('sub_prob/prob_lr_baseline.csv', index=None)
print('probation saved: ', time.time()-t1)
t1 = time.time()

# 生成提交结果
preds = np.argmax(preds, axis=1) # 返回概率最大的种类
test_pred = pd.DataFrame(preds, columns=['class'])
# test_pred.column = ['class']
test_pred['class'] = (test_pred['class']+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred['id'] = list(test_id)
test_pred[['id', 'class']].to_csv('sub/sub_lr_baseline.csv', index=None)
print('result generate done: ', time.time()-t1)
print('time use: ', time.time()-t0)