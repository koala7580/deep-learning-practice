"""MobileNet v2
论文：https://arxiv.org/pdf/1801.04381.pdf
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from layers import DepthwiseConv2D
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import xavier_initializer


IS_TRAINING = False
CHANNELS_AXIS = 1
DATA_FORMAT = 'channels_first'
L2_REQGULARIZER = l2_regularizer(0.01)
BATCH_NORMALIZATION_MOMENTUM = 0.995
BATCH_NORMALIZATION_EPSILON = 1e-5


def _batch_norm(x):
    return tf.layers.batch_normalization(x,
        axis=CHANNELS_AXIS, momentum=BATCH_NORMALIZATION_MOMENTUM,
        epsilon=BATCH_NORMALIZATION_EPSILON, training=IS_TRAINING, fused=True)


def _conv2d_with_bn(x, filters, kernel_size, strides, activation=tf.nn.relu6):
    x = tf.layers.conv2d(x, filters, kernel_size, strides, padding='same',
            kernel_regularizer=L2_REQGULARIZER,
            kernel_initializer=xavier_initializer(),
            data_format=DATA_FORMAT)
    x = _batch_norm(x)
    return activation(x)


def _depthwise_conv2d(x, kernel_size, strides):
    layer = DepthwiseConv2D(kernel_size, strides, data_format=DATA_FORMAT)
    return layer.apply(x)


def _bottleneck(inputs, filters, kernel_size, strides, t, r=True, activation=tf.nn.relu6):
    # data_format = channels_first
    input_dim = inputs.shape[CHANNELS_AXIS].value
    x = _conv2d_with_bn(inputs, t * input_dim, 1, 1)

    x = _depthwise_conv2d(x, kernel_size, strides)
    x = _batch_norm(x)
    x = activation(x)

    x = _conv2d_with_bn(x, filters, 1, 1)

    if r:
        x = tf.add(x, inputs)
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    x = _bottleneck(inputs, filters, kernel, strides, t, False)

    for _ in range(1, n):
        x = _bottleneck(x, filters, kernel, 1, t)

    return x


def build_model(inputs, args, mode, params):
    global IS_TRAINING
    IS_TRAINING = mode == tf.estimator.ModeKeys.TRAIN

    # Convert to channels_last format.
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

    scale_factor = 4

    x = _conv2d_with_bn(inputs, filters=32 // scale_factor, kernel_size=3, strides=2)

    x = _inverted_residual_block(x, 16 // scale_factor, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24 // scale_factor, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32 // scale_factor, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64 // scale_factor, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96 // scale_factor, (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160 // scale_factor, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320 // scale_factor, (3, 3), t=6, strides=1, n=1)

    x = _conv2d_with_bn(x, 1280, (1, 1), strides=(1, 1))

    x = tf.layers.average_pooling2d(x, pool_size=(7, 21), strides=1, data_format=DATA_FORMAT)
    x = tf.layers.flatten(x)
    x = tf.layers.dropout(x, 0.3)
    logits = tf.layers.dense(x, units=3)

    return logits
