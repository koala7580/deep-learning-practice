"""ShuffleNet
论文：https://arxiv.org/pdf/1707.01083.pdf
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import sys
import tensorflow as tf
from layers import DepthwiseConv2D
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import xavier_initializer


class ShuffleNet(object):

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

    def _group_conv2d(self, inputs, filters, kernel_size, groups):
        if self.data_format == 'channels_first':
            _, input_dim, h, w = inputs.get_shape()
        else:
            _, h, w, input_dim = inputs.get_shape()

        assert input_dim % groups == 0
        assert filters % groups == 0

        channels_per_group = input_dim // groups
        x = tf.reshape(inputs, [-1, groups, channels_per_group, h, w])

        def _conv2d_subgroup(x):
            x = tf.layers.conv2d(x, filters // groups,
                kernel_size=kernel_size, padding='same',
                kernel_initializer=xavier_initializer(),
                kernel_regularizer=self._l2_regularizer,
                data_format=self.data_format)
            return x

        if groups == 1:
            return _conv2d_subgroup(inputs)

        group_list = [_conv2d_subgroup(tf.identity(x[:, i, :, :, :])) for i in range(groups)]
        return tf.concat(group_list, self.channels_axis)

    def _channel_shuffle(self, inputs, groups):
        if self.data_format == 'channels_first':
            _, input_dim, h, w = inputs.get_shape()
            channels_per_group = input_dim // groups
            x = tf.reshape(inputs, [-1, groups, channels_per_group, h, w])
            x = tf.transpose(x, [0, 2, 1, 3, 4])
            x = tf.reshape(x, [-1, input_dim, h, w])
        else:
            _, h, w, input_dim = inputs.get_shape()
            channels_per_group = input_dim // groups
            x = tf.reshape(inputs, [-1, h, w, groups, channels_per_group])
            x = tf.transpose(x, [0, 1, 2, 4, 3])
            x = tf.reshape(x, [-1, h, w, input_dim])
        return x

    def _shufflenet_unit(self, inputs, filters, strides, groups, stage):
        groups = 1 if stage == 1 and inputs.get_shape()[self.channels_axis].value == 24 else groups

        x = self._group_conv2d(inputs, filters, kernel_size=1, groups=groups)
        x = self._activation(self._batch_norm(x))

        x = self._channel_shuffle(x, groups)

        x = self._depthwise_conv2d(x, 3, strides)
        x = self._batch_norm(x)

        x = self._group_conv2d(x, filters, kernel_size=1, groups=groups)
        x = self._batch_norm(x)

        if strides == 2:
            residual = tf.layers.average_pooling2d(inputs, 3, 2, padding='same', data_format=self.data_format)
            return tf.concat([x, residual], self.channels_axis)

        return self._activation(x + inputs)

    def __call__(self, inputs, training):
        self._is_training = training

        input_shape = inputs.get_shape()
        if self.data_format == 'channels_first' and input_shape[-1] == 3:
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        # scale_factor = 4

        x = self._conv2d_with_batch_norm(inputs, filters=24, kernel_size=3, strides=2)
        x = tf.layers.max_pooling2d(x, 3, 2, padding='same', data_format=self.data_format)

        groups = 3
        num_channels_per_layer = [240, 480, 960]
        num_layers_per_stage = [3, 7, 3]
        for stage in range(len(num_layers_per_stage)):
            x = self._shufflenet_unit(x, num_channels_per_layer[stage], 2, groups, stage + 1)
            for _ in range(num_layers_per_stage[stage]):
                x = self._shufflenet_unit(x, num_channels_per_layer[stage], 1, groups, stage + 1)

        x = tf.layers.average_pooling2d(x, pool_size=[7, 21], strides=1, data_format=self.data_format)
        logits = tf.layers.dense(x, units=2)
        return logits


def build_model(inputs, args, mode, params):
    model = ShuffleNet(data_format='channels_first')
    return model(inputs, mode == tf.estimator.ModeKeys.TRAIN)
