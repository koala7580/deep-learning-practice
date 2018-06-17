# -*- coding: utf-8 -*-
"""VGG model

论文地址：https://arxiv.org/pdf/1409.1556.pdf
参考文章：https://hackernoon.com/learning-keras-by-implementing-vgg16-from-scratch-d036733f2d5

结果记录：
2018-06-10 step=100000 loss = 0.73 accuracy=0.7412
"""
import tensorflow as tf
from estimators.base import BaseEstimator

class VGG16(BaseEstimator):
    """VGG model."""

    def __init__(self, is_training, data_format, batch_norm_decay, batch_norm_epsilon):
        """VGG constructor.
        Args:
        is_training: if build training or inference model.
        data_format: the data_format used during computation.
                    one of 'channels_first' or 'channels_last'.
        """
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._is_training = is_training
        assert data_format in ('channels_first', 'channels_last')
        self._data_format = data_format

    def build_model(self, x):
        if x.shape[1] == 3 or x.shape[1] == 4:
            input_data_format = 'channels_first'
        else:
            input_data_format = 'channels_last'

        x = self._transform_data_format(x, input_data_format, self._data_format)

        with tf.name_scope('group_1'):
            x = self._conv_bn(x, 64, 'conv1_1')
            x = self._conv_bn(x, 64, 'conv1_2')
            x = self._max_pool(x, 2, 2)

        with tf.name_scope('group_2'):
            x = self._conv_bn(x, 128, 'conv2_1')
            x = self._conv_bn(x, 128, 'conv2_2')
            x = self._max_pool(x, 2, 2)

        with tf.name_scope('group_3'):
            x = self._conv_bn(x, 256, 'conv3_1')
            x = self._conv_bn(x, 256, 'conv3_2')
            x = self._conv_bn(x, 256, 'conv3_3')
            x = self._max_pool(x, 2, 2)

        with tf.name_scope('group_4'):
            x = self._conv_bn(x, 512, 'conv4_1')
            x = self._conv_bn(x, 512, 'conv4_2')
            x = self._conv_bn(x, 512, 'conv4_3')
            x = self._max_pool(x, 2, 2)

        with tf.name_scope('group_5'):
            x = self._conv_bn(x, 512, 'conv5_1')
            x = self._conv_bn(x, 512, 'conv5_2')
            x = self._conv_bn(x, 512, 'conv5_3')
            x = self._max_pool(x, 2, 2)

        # Dense Layer
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 4096, activation=tf.nn.relu)
        x = tf.layers.dense(x, 4096, activation=tf.nn.relu)
        dropout = tf.layers.dropout(x, rate=0.4, training=self._is_training)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        return logits

    def _conv_bn(self, x, filters, name):
        x = tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same',
            name=name)

        tf.logging.info('image after unit %s: %s', x.name, x.get_shape())
        
        x = self._batch_norm(x, name='%s_bn' % name)

        return tf.nn.relu(x)


def build_model(input_layer, is_training, args, **kwargs):
    vgg = VGG16(is_training,
                args.data_format,
                args.batch_norm_decay,
                args.batch_norm_epsilon)
    return vgg.build_model(input_layer)
