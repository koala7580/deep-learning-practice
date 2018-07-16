"""SimpleNet
"""
from __future__ import division

import tensorflow as tf


class SimpleNet(object):

    def __init__(self, is_traning, data_format):
        self._is_training = is_traning
        self._data_format = data_format
        self._channels_axis = 1 if data_format == 'channels_first' else 3

    def _conv2d(self, x, filters, kernel_size, strides, **kwargs):
        x = tf.layers.conv2d(x, filters, kernel_size, strides,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                             padding='same', data_format=self._data_format, **kwargs)
        x = tf.layers.batch_normalization(x, epsilon=1e-5, momentum=0.995,
                                          training=self._is_training,
                                          fused=True)
        return self._activation(x)

    def _activation_v1(self, x):
        with tf.name_scope('polynomial_activation') as name_scope:
            with tf.variable_scope(name_scope):
                a1 = tf.get_variable('a1', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(0.01))
                a2 = tf.get_variable('a2', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(0.01))
                a3 = tf.get_variable('a3', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(1.00))

                tf.summary.scalar('%s/a1' % name_scope, a1)
                tf.summary.scalar('%s/a2' % name_scope, a2)
                tf.summary.scalar('%s/a3' % name_scope, a3)

        return a1 * x + a2 * x * x + a3 * x * x * x

    def _activation(self, x):
        return tf.nn.relu6(x)

    def _logits_conv2d(self, x):
        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        logits = tf.layers.conv2d(x, 10, 4, 1,
                                  kernel_regularizer=regularizer,
                                  data_format=self._data_format)
        return tf.layers.flatten(logits)

    def __call__(self, inputs):
        net = self._conv2d(inputs, 32, 3, 2)
        net = self._conv2d(net, 32, 3, 1)
        net = self._conv2d(net, 32, 3, 1)
        net = self._conv2d(net, 32, 3, 1)
        net = self._conv2d(net, 32, 3, 1)

        net = self._conv2d(net, 64, 3, 2)
        net = self._conv2d(net, 64, 3, 1)
        net = self._conv2d(net, 64, 3, 1)
        net = self._conv2d(net, 64, 3, 1)
        net = self._conv2d(net, 64, 3, 1)

        net = self._conv2d(net, 128, 3, 2)
        net = self._conv2d(net, 128, 3, 1)
        net = self._conv2d(net, 128, 3, 1)
        net = self._conv2d(net, 128, 3, 1)
        net = self._conv2d(net, 128, 3, 1)
        net = self._conv2d(net, 128, 3, 1)

        return self._logits_conv2d(net)


def build_model(inputs, args, mode, params):
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    net = SimpleNet(mode == tf.estimator.ModeKeys.TRAIN, 'channels_first')
    return net(inputs)
