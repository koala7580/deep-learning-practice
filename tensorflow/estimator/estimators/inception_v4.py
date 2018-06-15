# -*- coding: utf-8 -*-
"""Inception v4

论文地址：https://arxiv.org/pdf/1602.07261.pdf
参考文章：https://ai.googleblog.com/2016/08/improving-inception-and-image.html

这个模型太复杂了，先不实现了。
"""
import tensorflow as tf
from estimators.utils import build_model_fn


def conv2d_layer(inputs, filters, kernel_size, strides=1, **kwargs):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=tf.nn.relu, **kwargs)


def inception(inputs,
              conv_1x1_filters,
              conv_3x3_reduce_filters,
              conv_3x3_filters,
              conv_5x5_reduce_filters,
              conv_5x5_filters,
              pool_1x1_filters,
              name=''):
    # 1x1
    conv_1x1 = conv2d_layer(inputs, conv_1x1_filters, 1, name='%s_1x1' % name)

    # 3x3
    conv_3x3_reduce = conv2d_layer(inputs, conv_3x3_reduce_filters, 1, name='%s_3x3_reduce' % name)
    conv_3x3 = conv2d_layer(conv_3x3_reduce, conv_3x3_filters, 3, padding='same', name='%s_3x3' % name)

    # 5x5
    conv_5x5_reduce = conv2d_layer(inputs, conv_5x5_reduce_filters, 1, name='%s_5x5_reduce' % name)
    conv_5x5 = conv2d_layer(conv_5x5_reduce, conv_5x5_filters, 5, padding='same', name='%s_5x5' % name)

    # pool
    pool = tf.layers.max_pooling2d(inputs, 3, 1, padding='same', name='%s_pool' % name)
    pool_1x1 = conv2d_layer(pool, pool_1x1_filters, 1, name='%s_pool_1x1' % name)

    return tf.concat([conv_1x1, conv_3x3, conv_5x5, pool_1x1], 3)


def construct_model(input_layer, is_training, **kwargs):
    """Construct the model."""
    paddings1 = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    # paddings2 = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    paddings3 = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])

    conv1_7x7 = conv2d_layer(tf.pad(input_layer, paddings3), 64, 7, 2, name='conv1_7x7_s2')
    pool1 = tf.layers.max_pooling2d(tf.pad(conv1_7x7, paddings1), 3, strides=2)
    lrn1 = tf.nn.lrn(pool1, 5)

    conv2_3x3_reduce = conv2d_layer(lrn1, 64, 1, padding='same', name='conv2_3x3_reduce')
    conv2_3x3 = conv2d_layer(conv2_3x3_reduce, 192, 3, padding='same', name='conv2_3x3')
    pool2 = tf.layers.max_pooling2d(tf.pad(conv2_3x3, paddings1), 3, strides=2)

    inception_3a = inception(pool2, 64, 96, 128, 16, 32, 32, name='inception_3a')
    inception_3b = inception(inception_3a, 128, 128, 192, 32, 96, 64, name='inception_3b')

    pool3 = tf.layers.max_pooling2d(tf.pad(inception_3b, paddings1), 3, strides=2)

    inception_4a = inception(pool3, 192, 96, 208, 16, 48, 64, name='inception_4a')
    inception_4b = inception(inception_4a, 160, 112, 224, 24, 64, 64, name='inception_4b')
    inception_4c = inception(inception_4b, 128, 128, 256, 24, 64, 64, name='inception_4c')
    inception_4d = inception(inception_4c, 112, 144, 228, 32, 64, 64, name='inception_4d')
    inception_4e = inception(inception_4d, 256, 160, 320, 32, 128, 128, name='inception_4e')

    pool4 = tf.layers.max_pooling2d(tf.pad(inception_4e, paddings1), 3, strides=2)

    inception_5a = inception(pool4, 256, 160, 320, 32, 128, 128, name='inception_5a')
    inception_5b = inception(inception_5a, 384, 192, 384, 48, 128, 128, name='inception_5b')

    pool5 = tf.layers.average_pooling2d(inception_5b, 7, strides=1)


    # Dense Layer
    flatten = tf.layers.flatten(pool5)
    dropout = tf.layers.dropout(flatten, rate=0.4, training=is_training)
    dense = tf.layers.dense(dropout, units=1000, activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense, units=10)

    return logits


def build_estimator(config, params):
    """Build an estimator for train and predict.
    """
    return tf.estimator.Estimator(
        model_fn=build_model_fn(construct_model, resize_image=True),
        config=config,
        params=params
    )
