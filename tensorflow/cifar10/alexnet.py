"""AlexNet

论文地址：https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

结果：accuracy: 73 %
"""
from __future__ import division

import tensorflow as tf

def conv2d(x, filters, kernel_size, strides, **kwargs):
    return tf.layers.conv2d(x, filters, kernel_size, strides, padding="same", activation=tf.nn.elu, **kwargs)

def build_model(inputs, args, mode, params):
    scale_factor = 4

    with tf.name_scope('block_1'):
        x = conv2d(inputs, 96 // 4, 11, 2, name='conv1')
        x = tf.nn.lrn(x, 2, alpha=1e-4, beta=0.75, bias=1.0)
        x = tf.layers.max_pooling2d(x, 3, 2)

    with tf.name_scope('block_2'):
        x = conv2d(x, 256 // scale_factor, 5, 1, name='conv2')
        x = tf.nn.lrn(x, 2, alpha=1e-4, beta=0.75, bias=1.0)
        x = tf.layers.max_pooling2d(x, 3, 2)

    with tf.name_scope('block_3'):
        x = conv2d(x, 384 // scale_factor, 3, 1, name='conv3')

    with tf.name_scope('block_4'):
        x = conv2d(x, 384 // scale_factor, 3, 1, name='conv4')

    with tf.name_scope('block_5'):
        x = conv2d(x, 256 // scale_factor, 3, 1, name='conv5')
        x = tf.layers.max_pooling2d(x, 3, 2)

    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)
    x = tf.layers.dropout(x, args.dropout_rate)

    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)
    x = tf.layers.dropout(x, args.dropout_rate)

    logits = tf.layers.dense(x, 10)

    return logits
