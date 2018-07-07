"""Layers that are used to build CNN network.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

class LayerBuilder(object):

    def __init__(self,
        data_format='channels_last',
        is_training=False,
        batch_normalization_momentum=0.99,
        batch_normalization_epsilon=1e-3,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=None):
        self._data_format = data_format.lower()
        self._is_traning = is_training
        self._channel_index = 1 if self._data_format == 'channels_first' else 3
        self._data_format_1 = 'NCHW' if self._data_format == 'channels_firat' else 'NHWC'

        self._batch_normalization_momentum = batch_normalization_momentum
        self._batch_normalization_epsilon = batch_normalization_epsilon
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer


    def transform_data_format(self, x, from_data_format, to_data_format):
        assert from_data_format in ['channels_first', 'channels_last']
        assert to_data_format in ['channels_first', 'channels_last']

        if from_data_format != to_data_format:
            if from_data_format == 'channels_last':
                # Computation requires channels_first.
                x = tf.transpose(x, [0, 3, 1, 2])
            else:
                # Computation requires channels_last.
                x = tf.transpose(x, [0, 2, 3, 1])

        return x


    def conv2d(self, x, filters, kernel_size, strides, use_batch_normalization=True, **kwargs):
        x = tf.layers.conv2d(x, filters, kernel_size, strides,
                kernel_regularizer=self._kernel_regularizer,
                kernel_initializer=self._kernel_initializer, **kwargs)

        if use_batch_normalization:
            x = tf.layers.batch_normalization(x,
                axis=self._channel_index,
                momentum=self._batch_normalization_momentum,
                epsilon=self._batch_normalization_epsilon,
                trainable=False,
                training=self._is_traning,
                fused=True)

        return self._activation(x)

   def depthwise_conv2d(self, x, filters, kernel_size, strides):
        if type(kernel_size) == int:
           kernel_size = (kernel_size, kernel_size)
        
        if type(strides) == int:
            strides = (strides, strides)

        with tf.variable_scope('depthwise_conv2d'):
            in_channels = x.get_shape()[self._channel_index]

            W = tf.get_variable('kernel',
                shape=[kernel_size[0], kernel_size[1], in_channels, 1],
                dtype=tf.float32,
                initializer=self._kernel_initializer,
                regularizer=self._kernel_regularizer)
            b = tf.get_variable('biases',
                shape=[filters], dtype=tf.float32, initializer=tf.zeros_initializer())

            if self._data_format == 'channels_first':
                strides = [1, 1, strides[0], strides[1]]
            else:
                strides = [1, strides[0], strides[1], 1]

            y = tf.nn.depthwise_conv2d(x, W, strides, 'SAME', data_format=self._data_format_1) + b
        return y
