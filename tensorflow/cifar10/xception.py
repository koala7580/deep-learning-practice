"""Xception model
论文地址：https://arxiv.org/pdf/1610.02357.pdf

结果：accuracy:
"""
from __future__ import division

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

class Builder(object):

    def __init__(self, is_training, data_format):
        self.is_traning = is_training
        self.data_format = data_format

    def batch_norm(self, inputs):
        """Performs a batch normalization using a standard set of parameters."""
        # We set fused=True for a significant performance boost. See
        # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        return tf.layers.batch_normalization(
            inputs=inputs, axis=1 if self.data_format == 'channels_first' else 3,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=self.is_traning, fused=True)

    def conv2d_bn(self, x, filters, kernel_size, strides=1):
        x = tf.layers.conv2d(x, filters, kernel_size, strides, padding='same', data_format=self.data_format)
        x = self.batch_norm(x)
        return tf.nn.relu(x)

    def residual(self, x, filters):
        x = tf.layers.conv2d(x, filters, kernel_size=1, strides=2, padding='same', data_format=self.data_format)
        return self.batch_norm(x)

    def separable_conv2d_bn(self, x, filters):
        x = tf.layers.separable_conv2d(x, filters, 3, padding='same', data_format=self.data_format)
        return self.batch_norm(x)

    def max_pool(self, x):
        return tf.layers.max_pooling2d(x, 2, 2)


def build_model(inputs, args, mode, params):
    scale_factor = 1

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    builder = Builder(is_training, 'channels_last')

    # Entry Flow
    net = builder.conv2d_bn(inputs, 32 // scale_factor, 3, 2)
    net = builder.conv2d_bn(net, 64 // scale_factor, 3)
    residual = builder.residual(net, 128 // scale_factor)

    net = builder.separable_conv2d_bn(net, 128 // scale_factor)
    net = tf.nn.relu(net)
    net = builder.separable_conv2d_bn(net, 128 // scale_factor)
    net = builder.max_pool(net)
    net = net + residual

    residual = builder.residual(net, 256 // scale_factor)
    net = builder.separable_conv2d_bn(tf.nn.relu(net), 256 // scale_factor)
    net = builder.separable_conv2d_bn(tf.nn.relu(net), 256 // scale_factor)
    net = builder.max_pool(net)

    net = net + residual

    residual = builder.residual(net, 728 // scale_factor)
    net = builder.separable_conv2d_bn(tf.nn.relu(net), 728 // scale_factor)
    net = builder.separable_conv2d_bn(tf.nn.relu(net), 728 // scale_factor)
    net = builder.max_pool(net)
    net = net + residual

    # Middle Flow
    for _ in range(8 // scale_factor):
        residual = builder.residual(net, 728 // scale_factor)
        net = builder.separable_conv2d_bn(tf.nn.relu(net), 728 // scale_factor)
        net = builder.separable_conv2d_bn(tf.nn.relu(net), 728 // scale_factor)
        net = builder.separable_conv2d_bn(tf.nn.relu(net), 728 // scale_factor)
        net = net + residual

    # Exit Flow
    residual = builder.residual(net, 1024 // scale_factor)
    net = builder.separable_conv2d_bn(tf.nn.relu(net), 728 // scale_factor)
    net = builder.separable_conv2d_bn(tf.nn.relu(net), 1024 // scale_factor)
    net = builder.max_pool(net)


    net = net + residual

    net = builder.separable_conv2d_bn(net, 1536 // scale_factor)
    net = builder.separable_conv2d_bn(tf.nn.relu(net), 2048 // scale_factor)

    net = tf.layers.average_pooling2d(net, 1, 1)
    net = tf.layers.flatten(net)
    logits = tf.layers.dense(net, units=10)

    return logits
