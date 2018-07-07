"""MobileNet v1 model
论文地址：https://arxiv.org/pdf/1704.04861.pdf

结果：accuracy:
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.995
_BATCH_NORM_EPSILON = 1e-5

class Builder(object):

    def __init__(self, is_training, data_format):
        self.is_traning = is_training
        self.data_format = data_format
        self.weight_decay = 0.01
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
        self.xavier_initializer = tf.contrib.layers.xavier_initializer()
        self.channel_index = 1 if self.data_format == 'channels_first' else 3

    def _activation(self, x):
        return tf.nn.elu(x)

    def _batch_norm(self, inputs):
        """Performs a batch normalization using a standard set of parameters."""
        # We set fused=True for a significant performance boost. See
        # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        return tf.layers.batch_normalization(
            inputs=inputs,
            axis=self.channel_index,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
            center=True, scale=True, training=self.is_traning, fused=True)

    def conv2d_bn(self, x, filters, kernel_size, strides=1):
        x = tf.layers.conv2d(
            x, filters, kernel_size, strides,
            padding='same', data_format=self.data_format,
            kernel_regularizer=self.l2_regularizer,
            kernel_initializer=self.xavier_initializer)
        x = self._batch_norm(x)
        return self._activation(x)

    def depthwise_conv2d(self, x, filters, kernel_size, strides):
        with tf.variable_scope('depthwise_conv2d'):
            input_shape = x.get_shape()
            n_channels = input_shape[self.channel_index]
            W = tf.get_variable('weights',
                shape=[kernel_size, kernel_size, n_channels, 1],
                dtype=tf.float32, initializer=self.xavier_initializer)
            b = tf.get_variable('biases',
                shape=[filters], dtype=tf.float32, initializer=tf.zeros_initializer())

            if self.data_format == 'channels_first':
                strides = [1, 1, strides, strides]
            else:
                strides = [1, strides, strides, 1]

            data_format = 'NCHW' if self.data_format == 'channels_first' else 'NHWC'
            y = tf.nn.depthwise_conv2d(x, W, strides, 'SAME', data_format=data_format) + b
        return y

    def separable_conv2d(self, x, filters, strides):
        with tf.name_scope('separable_conv2d') as name_scope:
            with tf.variable_scope(name_scope):
                y = self.depthwise_conv2d(x, x.get_shape()[self.channel_index], 3, strides)
                y = self._batch_norm(y)
                y = self._activation(y)
                y = tf.layers.conv2d(y, filters, 1, 1, padding='same')
                y = self._batch_norm(y)
                y = self._activation(y)
        return y


def build_model(inputs, args, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    B = Builder(is_training, 'channels_last')

    net = B.conv2d_bn(inputs, 32, 3, 2)   # 16
    net = B.separable_conv2d(net, 64, 1)  # 16
    net = B.separable_conv2d(net, 128, 2) # 8
    net = B.separable_conv2d(net, 128, 1) # 8
    net = B.separable_conv2d(net, 256, 2) # 4
    net = B.separable_conv2d(net, 256, 1) # 4
    net = B.separable_conv2d(net, 512, 1) # 4 here should downsample
    net = B.separable_conv2d(net, 512, 1) # 4 here should downsample
    net = B.separable_conv2d(net, 512, 1) # 4 here should downsample
    net = B.separable_conv2d(net, 512, 1) # 4 here should downsample

    net = tf.layers.average_pooling2d(net, 4, 1)
    net = tf.layers.flatten(net)
    logits = tf.layers.dense(net, units=10)

    return logits
