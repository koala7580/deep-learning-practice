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


class MobileNetV2(object):

    def __init__(self,
                 data_format='channels_last',
                 weight_decay=0.01,
                 batch_norm_momentum=0.995,
                 batch_norm_epsilon=1e-5):
        self._is_training = False
        self.data_format = data_format
        self._l2_regularizer = l2_regularizer(weight_decay)
        self._batch_norm_epsilon = batch_norm_epsilon
        self._batch_norm_momentum = batch_norm_momentum

    @property
    def channels_axis(self):
        return 1 if self.data_format == 'channels_first' else -1

    def _activation(self, x):
        return tf.nn.relu6(x)

    def _batch_norm(self, x):
        return tf.layers.batch_normalization(x,
            axis=self.channels_axis, momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon, training=self._is_training, fused=True)

    def _conv2d_with_batch_norm(self, x, filters, kernel_size, strides):
        x = tf.layers.conv2d(x, filters, kernel_size, strides, padding='same',
            kernel_regularizer=self._l2_regularizer,
            kernel_initializer=xavier_initializer(),
            data_format=self.data_format)
        return self._activation(self._batch_norm(x))

    def _depthwise_conv2d(self, x, kernel_size, strides):
        layer = DepthwiseConv2D(kernel_size, strides, padding='same', data_format=self.data_format)
        return layer.apply(x)

    def _bottleneck(self, inputs, filters, kernel_size, strides, t, r=True, activation=tf.nn.relu6):
        input_dim = inputs.shape[self.channels_axis].value
        x = self._conv2d_with_batch_norm(inputs, t * input_dim, 1, 1)

        x = self._depthwise_conv2d(x, kernel_size, strides)
        x = self._activation(self._batch_norm(x))

        x = self._conv2d_with_batch_norm(x, filters, 1, 1)

        if r:
            x = tf.add(x, inputs)
        return x

    def _inverted_residual_block(self, inputs, filters, kernel, t, strides, n):
        x = self._bottleneck(inputs, filters, kernel, strides, t, False)

        for _ in range(1, n):
            x = self._bottleneck(x, filters, kernel, 1, t)

        return x

    def __call__(self, inputs, training):
        self._is_training = training

        input_shape = inputs.get_shape()
        if self.data_format == 'channels_first' and input_shape[-1] == 3:
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        scale_factor = 4

        x = self._conv2d_with_batch_norm(inputs, filters=32 // scale_factor, kernel_size=3, strides=2)

        kernel_size = 3

        x = self._inverted_residual_block(x, 16 // scale_factor, kernel_size, t=1, strides=1, n=1)
        x = self._inverted_residual_block(x, 24 // scale_factor, kernel_size, t=6, strides=2, n=2)
        x = self._inverted_residual_block(x, 32 // scale_factor, kernel_size, t=6, strides=2, n=3)
        x = self._inverted_residual_block(x, 64 // scale_factor, kernel_size, t=6, strides=2, n=4)
        x = self._inverted_residual_block(x, 96 // scale_factor, kernel_size, t=6, strides=1, n=3)
        x = self._inverted_residual_block(x, 160 // scale_factor, kernel_size, t=6, strides=2, n=3)
        x = self._inverted_residual_block(x, 320 // scale_factor, kernel_size, t=6, strides=1, n=1)

        x = self._conv2d_with_batch_norm(x, 1280 // scale_factor, (1, 1), strides=(1, 1))

        x = tf.layers.average_pooling2d(x, pool_size=(7, 21), strides=1, data_format=self.data_format)
        x = tf.layers.flatten(x)
        x = tf.layers.dropout(x, 0.3)
        return tf.layers.dense(x, units=3)


def build_model(inputs, args, mode, params):
    mobilenet = MobileNetV2(data_format='channels_first')
    return mobilenet(inputs, mode == tf.estimator.ModeKeys.TRAIN)
