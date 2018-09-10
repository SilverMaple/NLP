# -*- coding: utf-8 -*-
# @Time    : 2018/8/9 16:01
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : cnn_model.py

import tensorflow as tf
import logging


class TCNNConfig(object):
    '''CNN配置参数'''

    embedding_dim = 128 # 词向量维度
    seq_length = 4000 # 序列长度，即每篇文章中词的数量
    num_classes = 19 # 类别数 19+1, 因为从1开始
    num_filters = 128 # 卷积核数
    # kernel_size = 5 # 卷积核尺寸
    # kernel_size = [2, 3, 4, 5] # 卷积核尺寸，在自然语言处理时相当于对one-hot向量提取n-gram特征
    kernel_size = [5] # 卷积核尺寸，在自然语言处理时相当于对one-hot向量提取n-gram特征
    context_size = 1500000 # 文章数量?102277

    hidden_dim = 128 # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3 # 学习率

    batch_size = 64 # 每次训练大小
    num_epochs = 5 # 迭代次数

    print_per_batch = 10
    save_per_batch = 10 # 存入tensorBoard轮数

class TextCNN(object):
    '''文本分类CNN模型'''

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.float32, [None, None, self.config.embedding_dim], name='input_x')
        # self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') # 神经元被选中概率

        self.cnn()

    def cnn(self):
        '''CNN模型'''
        # 词向量映射，指定运行设备为cpu
        # with tf.device('/cpu:0'):
        #     embedding = tf.get_variable('embedding', [self.config.context_size, self.config.embedding_dim])
        #     embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)


        pooled_output = []
        for i, filter_size in enumerate(self.config.kernel_size):
            with tf.name_scope('conv-maxpool-%s'%filter_size):
                # 卷积层，提取n-gram特征
                conv = tf.layers.conv1d(self.input_x, self.config.num_filters,
                                        filter_size, name='conv-%s'%filter_size)
                # # 全局最大池化层，global max polling layer
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

                # 输入为最大池化层，由全连接层神经元处理
                fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1-%s'%filter_size)
                # dropout，防止或减轻过拟合，由keep_prob控制丢弃概率
                fc = tf.contrib.layers.dropout(fc, self.keep_prob)
                # 激活函数，引入非线性因素，负数直接变为0.0，正数保持不变
                # ReLU—平滑函数→softplus—求微分→logistic—泛化→softmax
                fc = tf.nn.relu(fc)
                pooled_output.append(fc)
        h_pool = tf.concat(pooled_output, 1)  # concat on 2rd dimension
        filter_total = 128
        # filter_total = self.config.num_filters * max(self.config.kernel_size)
        self.h_pool_flat = tf.reshape(h_pool, [-1, filter_total], name='h_pool_flat')

        with tf.name_scope('dropout-output'):
            # fc = tf.nn.dropout(self.h_pool_flat, self.keep_prob, name="dropout")
            # 分类器
            # 大小为[batchsize，num_classes]
            # self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.logits = tf.layers.dense(self.h_pool_flat, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1) # 寻找最大值预测类别

        with tf.name_scope('optimize'):
            # 损失函数，交叉熵
            # 先对网络最后一层做softmax获得类别概率向量，再与实际标签做交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            # 对向量求均值，获得loss
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            # 寻找全局最优点的优化算法，引入了二次方梯度校正
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


