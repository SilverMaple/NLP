# -*- coding: utf-8 -*-
# @Time    : 2018/8/9 17:14
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : predict.py

from __future__ import print_function

import os
import sys
sys.path.append('drive/DGNLP')
sys.path.append('..')
import logging
import tensorflow as tf
import tensorflow.contrib.keras as kr

from text_cnn.cnn_model import TCNNConfig, TextCNN
from text_cnn.data_util import read_file

base_dir = 'drive/DGNLP/nlp-data'
test_set_file = os.path.join(base_dir, 'test_set.csv')
save_dir = 'drive/DGNLP/text_cnn/checkpoints/'
save_path = os.path.join(save_dir, 'best_validation')
result_path = os.path.join(save_dir, 'result.csv')

class CnnModel:

    def __init__(self):
        self.config = TCNNConfig()
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)

    def predict(self, message):
        data = message.split(' ')
        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return y_pred_cls

if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)  # 显示INFO等级以上日志

    cnn_model = CnnModel()
    logging.info('Start reading test_set...')
    test_set = read_file(test_set_file)
    logging.info('Reading test_set done.')
    test_set_data = test_set['data']
    test_set_id = test_set['id']
    logging.info('Start predicting and writing result...')
    fid0 = open(result_path, 'w')
    fid0.write('id,class' + '\n')
    for i in range(len(test_set_data)):
        c = cnn_model.predict(test_set_data[i])
        fid0.write(str(test_set_id[i]) + ',' + str(c) + '\n')
    fid0.close()
    logging.info('All done.')

