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
        input_data_format = self._detect_data_format(x)
        x = self._transform_data_format(x, input_data_format, self._data_format)

        filters = [32, 64, 128, 256, 256]
        layers = [2, 2, 3, 3, 3]


        for i in range(5):
            with tf.name_scope('stage') as name_scope:
                for j in range(layers[i]):
                    x = self._conv_bn(x, filters[i], 'conv%d_%d' % (i, j))
                x = self._max_pool(x, 2, 2)

                tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())

        # Dense Layer
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 2048, activation=tf.nn.relu)
        x = tf.layers.dense(x, 2048, activation=tf.nn.relu)
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

        x = self._batch_norm(x)

        return tf.nn.relu(x)


def build_model(input_layer, is_training, args, **kwargs):
    vgg = VGG16(is_training,
                args.data_format,
                args.batch_norm_decay,
                args.batch_norm_epsilon)
    return vgg.build_model(input_layer)
