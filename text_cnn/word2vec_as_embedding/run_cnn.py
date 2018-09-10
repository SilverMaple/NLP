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
import tensorflow as tf
from sklearn import metrics

from numba.decorators import jit
from text_cnn.cnn_model import TCNNConfig, TextCNN
from utils.word2vec import Manager
from text_cnn.data_util import batch_iter, process_file
from utils.commons import SUPERPARAMS
batch_iter_jit = batch_iter
# batch_iter_jit = jit(batch_iter)

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
    batch_eval = batch_iter_jit(x_, y_, wv_model, config.batch_size)
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
    return manager.load_model("cache/train2.model")

def train():
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
    init = tf.global_variables_initializer()
    conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=conf)
    session.run(init)
    writer.add_graph(session.graph)

    logging.info('Training and evaluating...')
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 1000

    flag = False
    wv_model = get_wordvec_model()
    for epoch in range(config.num_epochs):
        logging.info('Epoch: {}'.format(epoch + 1,))
        batch_train = batch_iter_jit(x_train, y_train, wv_model, config.batch_size)
        # batch_train = batch_iter(x_train, y_train, config.batch_size)
        logging.info('batch data fetch success.')
        # logging.info(type(batch_train))
        for x_batch, y_batch in batch_train:
            # logging.info('{}: x_batch length: {}'.format(total_batch, len(x_batch)))
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 训练结果写入tensorboard轮数
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 输出训练集和验证集性能轮数
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
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.8}, Train Acc: {2:>7.8%},' \
                      'Val Loss: {3:>6.8}, Val Acc: {4:>7.8%}, Time: {5} {6}'
                logging.info(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val,
                                        time_dif, improved_str))

            # 运行优化
            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                logging.info('No optimization for a long time, auto-stopping...')
                flag = True
                break

        if flag:
            break

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
    wv_model = get_wordvec_model()
    for epoch in range(1, config.num_epochs+1):
        logging.info('Epoch: {}'.format(epoch))
        batch_train = batch_iter_jit(x_train, y_train, wv_model, config.batch_size)
        # batch_train = batch_iter(x_train, y_train, config.batch_size)
        # logging.info('batch data fetch success.')
        # logging.info(type(batch_train))
        total_batch = 0
        for x_batch, y_batch in batch_train:
            # logging.info('{}: x_batch length: {}'.format(total_batch, len(x_batch)))
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.print_per_batch == 0:
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                time_dif = get_time_dif(start_time)
                logging.info('epoch:{4: >3}, Iter: {0:>6}, Train Loss: {1:>6.8}, Train Acc: {2:>7.8%}, Time: {3}'
                             .format(total_batch, loss_train, acc_train, time_dif, epoch))

            # 运行优化
            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1

        # 所有batch结束，用验证集验证
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
            epoch_name = os.path.join(save_dir, "epoch_{0}".format(epoch))
            saver.save(sess=session, save_path=epoch_name)
    session.close()

def test():
    logging.info('Loading test data...')
    start_time = time.time()
    x_test, y_test = process_file(test_set_file)

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


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)  # 显示INFO等级以上日志

    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    logging.info('Configuring CNN model...')
    config = TCNNConfig()
    # if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    #     build_voca b(train_dir, vocab_dir, config.vocab_size)
    # categories, cat_to_id = read_category()
    # words, word_to_id = read_vocab(vocab_dir)
    # config.context_size = len(words)
    model = TextCNN(config)
    if sys.argv[1] == 'train':
        train2(restore=False)
    else:
        test()
