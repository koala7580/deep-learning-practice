"""MobileNet v2 model
论文地址：https://arxiv.org/pdf/1801.04381.pdf

结果：accuracy:
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from layers import LayerBuilder


def build_model(inputs, args, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    B = LayerBuilder(
        is_training=is_training,
        data_format='channels_last',
        batch_normalization_momentum=0.995,
        batch_normalization_epsilon=1e-5,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))

    inputs = B.transform_data_format(inputs, 'channels_last', 'channels_first')

    net = B.conv2d(inputs, 32, 3, 2)   # 16

    net = tf.layers.average_pooling2d(net, 4, 1)
    net = tf.layers.flatten(net)
    logits = tf.layers.dense(net, units=10)

    return logits
