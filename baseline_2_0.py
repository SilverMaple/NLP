# -*- coding: utf-8 -*-
# @Time    : 2018/7/29 15:59
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : baseline_2_0.py

import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
import time

print('Start...')
t0 = time.time()
t1 = time.time()
column = 'word_seg'
train_set = pd.read_csv('input/train_set.csv')
print('train_set.csv read done: ', time.time()-t1)
t1 = time.time()
test_set = pd.read_csv('input/test_set.csv')
print('test_set.csv read done: ', time.time()-t1)
t1 = time.time()
test_id = test_set['id'].copy

vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9,
                      use_idf=1, smooth_idf=1, sublinear_tf=1)
train_set_term_doc = vec.fit_transform(train_set[column])
test_set_term_doc = vec.transform(test_set[column])

y = (train_set['class']-1).astype(int)
lin_clf = svm.LinearSVC()
lin_clf.fit(train_set_term_doc, y)
print('train done: ', time.time()-t1)
t1 = time.time()
preds = lin_clf.predict(test_set_term_doc)
print('prediction done: ', time.time()-t1)
t1 = time.time()

i = 0
fid0 = open('baseline.csv', 'w')
fid0.write('id,class'+'\n')
for item in preds:
    fid0.write(str(i) + ',' + str(item+1)+'\n')
    i = i+1
fid0.close()
print('result generate done: ', time.time()-t1)
print('time use: ', time.time()-t0)