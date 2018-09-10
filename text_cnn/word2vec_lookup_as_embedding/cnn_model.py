# -*- coding: utf-8 -*-
# @Time    : 2018/8/9 16:01
# @Author  : SilverMaple
# @Site    : https://github.com/SilverMaple
# @File    : cnn_model.py

import tensorflow as tf


class TCNNConfig(object):
    '''CNN配置参数'''

    embedding_dim = 256 # 词向量维度?100
    seq_length = 1500 # 序列长度，即每篇文章中词的数量4000?
    num_classes = 19 # 类别数 19+1, 因为从1开始
    num_filters = 128 # 卷积核数256
    # kernel_size = 5 # 卷积核尺寸
    filter_sizes = [2, 3, 4, 5, 6]  # 卷积核尺寸，在自然语言处理时相当于对one-hot向量提取n-gram特征
    context_size = 102277 # 文章数量?102277?2002277

    hidden_dim = 64 # 全连接层神经元128

    dropout_keep_prob = 0.6 # dropout保留比例
    learning_rate = 1e-3 # 学习率
    # learning_rate = 1e-3 # 学习率

    # 在卷积神经网络的学习过程中，小批次会表现得更好，选取范围一般位于区间[16, 128]内
    batch_size = 64 # 每次训练大小
    num_epochs = 10 # 迭代次数

    print_per_batch = 10
    save_per_batch = 10 # 存入tensorBoard轮数
    save_per_epoch = 1


class TextCNN(object):
    '''文本分类CNN模型'''

    def __init__(self, config, vocab_size=None, embedding_layer_w=None):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')

        # self.input_x = tf.placeholder(tf.float32, [None, None, self.config.embedding_dim], name='input_x')
        # self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') # 神经元被选中概率

        self.cnn(vocab_size=vocab_size, embedding_layer_w=embedding_layer_w)

    def cnn(self, vocab_size=None, embedding_layer_w=None, restore=True):
        '''CNN模型'''
        # 词向量映射，指定运行设备为cpu
        # with tf.device('/cpu:0'):
        #     embedding = tf.get_variable('embedding', [self.config.context_size, self.config.embedding_dim])
        #     embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        # embedding_inputs = self.input_x
        l2_loss = tf.constant(0.0)
        l2_reg_lambda = 0.0

        # # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if restore == False:
                self.W = tf.Variable(tf.convert_to_tensor(embedding_layer_w), name="W")
            else:
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, self.config.embedding_dim], -1.0, 1.0),
                    name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.config.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("optimzation"):
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
            # self.optim = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

