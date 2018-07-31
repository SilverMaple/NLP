# -*- coding: utf-8 -*-
# @Time    : 2018/7/30 23:59
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : baseline_3_0.py

import pandas as pd
from pyfasttext import FastText
import time

print('Start...')
t0 = time.time()
t1 = time.time()
# terminal command
# fasttext/fastText-0.1.0/fasttext supervised -input /media/sf_NLP/input/train_set.txt -output /media/sf_NLP/input/fasttext.model -label __label__ -lr 1.0 -minCount 3 -wordNgrams 1,2 -dim 2000
#classifier = fasttext.supervised("/media/sf_NLP/input/train_set.txt","/media/sf_NLP/input/#fasttext.model",label_prefix="__label__")
#print('train_use: ', time.time()-t1)
#t1 = time.time()

# loadModel
# classifier = fasttext.load_model('/media/sf_NLP/input/fasttext.model.bin', label_prefix='__label__')
classifier = FastText('/media/sf_NLP/input/fasttext.model.bin')

# test
# result = classifier.test("/media/sf_NLP/input/test_set.txt")
with open("/media/sf_NLP/input/test_set.txt") as fr:
    lines = fr.readlines()
labels_predict = [e[0] for e in classifier.predict(lines)] #预测输出结果为二维形式
# labels_predict = [e[0][0] for e in classifier.predict_proba(lines)]
print('prediction done: ', time.time()-t1)
t1 = time.time()

i = 0
fid0 = open('baseline.csv', 'w')
fid0.write('id,class'+'\n')
for item in labels_predict:
    fid0.write(str(i) + ',' + str(item)+'\n')
    i = i+1
fid0.close()
print('result generate done: ', time.time()-t1)
print('time use: ', time.time()-t0)
