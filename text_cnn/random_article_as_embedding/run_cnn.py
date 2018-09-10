# -*- coding: utf-8 -*-
# @Time    : 2018/8/9 17:14
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : run_cnn.py

from __future__ import print_function

import os
import sys
sys.path.append('drive/DGNLP')
sys.path.append('..')

import time
import logging
from datetime import timedelta
import numpy as np
import tensorflow.contrib.keras as kr

import pandas as pd
import tensorflow as tf
from sklearn import metrics

from numba.decorators import jit
from text_cnn.cnn_model import TCNNConfig, TextCNN
from utils.word2vec import Manager
from text_cnn.data_util import batch_iter, process_file
from utils.commons import SUPERPARAMS

base_dir = 'nlp-data'
train_set_file = os.path.join(base_dir, 'train_set.csv')
test_set_file = os.path.join(base_dir, 'test_set.csv')
validate_set_file = os.path.join(base_dir, 'validate_set.csv')

save_dir = 'text_cnn/checkpoints/'
save_path = os.path.join(save_dir, 'best_validation')

def get_time_dif(start_time):
    '''获取使用时间'''
    time_dif = time.time() - start_time
    return time_dif
    # return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_, wv_model=None):
    '''评估数据的准确率和损失'''
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, wv_model, config.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss*batch_len
        total_acc += acc*batch_len

    return total_loss / data_len, total_acc / data_len

def get_wordvec_model():
    # global max_num
    max_num = SUPERPARAMS.WV_MAX_NUM
    min_count = SUPERPARAMS.WV_MIN_COUNT
    size = SUPERPARAMS.WV_SIZE
    window = SUPERPARAMS.WV_WINDOW
    epoch = SUPERPARAMS.WV_EPOCH
    learning_rate = SUPERPARAMS.WV_LEARNING_RATE
    manager = Manager(retrain=True, min_count=min_count, size=size, window=window, epoch=epoch,
                      learning_rate=learning_rate)
    return manager.load_model("cache/train3.model")

def train2(restore=False):
    logging.info('Configuring TensorBoard and Saver...')
    # 配置tensor board
    tensorboard_dir = 'text_cnn/tmp'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    #配置Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.info('Loading training and validation data...')
    # 载入训练集与验证集
    start_time = time.time()
    train_set_data = process_file(train_set_file, config.num_classes, config.seq_length)
    x_train, y_train = train_set_data['train_set']
    x_val, y_val = train_set_data['validate_set']
    del train_set_data
    logging.info('Time usage: {}'.format(get_time_dif(start_time,)))

    # 创建session
    conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=conf)
    if restore and os.path.exists(save_dir+"checkpoint"):
        logging.info("Restoring Variables from Checkpoint for cnn model.")
        saver.restore(session, tf.train.latest_checkpoint(save_dir))
    else:
        logging.info('first training cnn model, Initializing Variables')
        session.run(tf.global_variables_initializer())

    writer.add_graph(session.graph)

    logging.info('Training and evaluating...')
    start_time = time.time()

    best_acc_val = 0.0

    flag = False
    # wv_model = get_wordvec_model()
    for epoch in range(1, config.num_epochs+1):
        logging.info('Epoch: {}'.format(epoch))
        batch_train = batch_iter(x_train, y_train, wv_model, config.batch_size)
        total_batch = 0
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.print_per_batch == 0:
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                time_dif = get_time_dif(start_time)
                logging.info('epoch:{4: >3}, Iter: {0:>6}, Train Loss: {1:>6.8}, Train Acc: {2:>7.8%}, Time: {3}'
                             .format(total_batch, loss_train, acc_train, time_dif, epoch))

            # 运行优化
            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1

        feed_dict[model.keep_prob] = 1.0
        loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
        loss_val, acc_val = evaluate(session, x_val, y_val, wv_model=wv_model)

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            last_improved = total_batch
            saver.save(sess=session, save_path=save_path)
            improved_str = '*'
        else:
            improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = 'epoch:{0: >3}, Iter: {1:>6}, Train Loss: {2:>6.8}, Train Acc: {3:>7.8%},' \
              'Val Loss: {4:>6.8}, Val Acc: {5:>7.8%}, Time: {6} {7}'
        logging.info(msg.format(epoch, total_batch, loss_train, acc_train, loss_val, acc_val,
                                time_dif, improved_str))

        if epoch % config.save_per_epoch == 0:
            s = session.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, total_batch)
            # epoch_name = os.path.join(save_dir, "epoch_{0}".format(epoch))
            # saver.save(sess=session, save_path=epoch_name)
    session.close()

def test():
    logging.info('Loading test data...')
    start_time = time.time()
    x_test = None
    y_test = None
    # x_test, y_test = process_file(test_set_file)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
    logging.info('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    logging.info(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    logging.info("Precision, Recall and F1-Score...")
    logging.info(metrics.classification_report(y_test_cls, y_pred_cls))

    # 混淆矩阵
    logging.info("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    logging.info(cm)

    time_dif = get_time_dif(start_time)
    logging.info("Time usage:", time_dif)
    
def generateDataMatrix():
    train_set = pd.read_csv(train_set_file)['word_seg']
    logging.info(len(train_set))
    logging.info('Loading word2vec model')
    wv_model = get_wordvec_model()
    config = TCNNConfig()
    x_shuffle_split = np.array([i.split(' ') for i in train_set])
    for i, x in enumerate(x_shuffle_split):
        if i % 100 == 0:
            logging.info('{}...'.format(i))
        x_shuffle_split[i] = [wv_model.wv.vocab[word].index for word in x if word in wv_model.wv.vocab]
    logging.info('word2id done.')
    x_shuffle_split = kr.preprocessing.sequence.pad_sequences(x_shuffle_split, config.seq_length)
    logging.info('pad done.')
    np.save("cache/data_matrix.npy", x_shuffle_split)
    logging.info('save done.')
    
def random_vector_generate(filename):
    # for i in wv_model_vectors.shape[0]:
    #     wv_model_vectors[i] = np.random.randn(128)
    wv_model_vectors = np.random.randn(wv_model_vectors.shape[0], wv_model_vectors.shape[1])
    np.save('random_embedding.npy', wv_model_vectors)
    logging.info('Saved random embedding vectors.')

if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)  # 显示INFO等级以上日志

    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    logging.info('Configuring CNN model...')
    config = TCNNConfig()
    wv_model = get_wordvec_model()
    # 随机初始化词向量
    wv_model_vectors = np.random.randn(len(wv_model.wv.vectors), config.embedding_dim)
    np.save('random_embedding.npy', wv_model_vectors)
    logging.info('Saved random embedding vectors.')
    embedding_layer_w = np.row_stack((wv_model_vectors.astype('float32'), np.zeros(shape=(config.embedding_dim)).astype('float32')))
    # embedding_layer_w = np.row_stack((wv_model.wv.vectors, np.zeros(shape=(config.embedding_dim)).astype('float32')))
    logging.info('embedding_layer_w shape: {}'.format(embedding_layer_w.shape))
    # embedding_layer_w = np.load("cache/train2.model.wv.vectors.npy").astype('float32')
    vocab_size = embedding_layer_w.shape[0]
    logging.info('vocab_size:{}'.format(vocab_size))
    model = TextCNN(config, vocab_size=vocab_size, embedding_layer_w=embedding_layer_w)
    if sys.argv[1] == 'train':
        train2(restore=False)
        # generateDataMatrix()
    else:
        # generateDataMatrix()
        test()
