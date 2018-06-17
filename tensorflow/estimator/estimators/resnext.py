# -*- coding: utf-8 -*-
"""ResNeXt

论文地址：https://arxiv.org/pdf/1611.05431.pdf
参考文章：https://blog.csdn.net/u014380165/article/details/71667916

结果记录：
"""
import tensorflow as tf
from estimators.base import BaseEstimator

class ResNeXt(BaseEstimator):
    """ResNeXt model."""

    def __init__(self, is_training, data_format, batch_norm_decay, batch_norm_epsilon):
        """ResNet constructor.
        Args:
        is_training: if build training or inference model.
        data_format: the data_format used during computation.
                    one of 'channels_first' or 'channels_last'.
        """
        super(ResNeXt, self).__init__(
            is_training,
            data_format,
            batch_norm_decay,
            batch_norm_epsilon)

    def _get_layers(self, num_layers):
        if num_layers == 18:
            layers = [1, 1, 1, 1]
        elif num_layers == 34:
            layers = [2, 3, 5, 2]
        elif num_layers == 50:
            layers = [2, 3, 5, 2]
        else:
            raise ValueError('%d layers is not supportted', num_layers)

        pad = None
        block_func = self._build_block
        return layers, pad, block_func

    def build_model(self, x, num_layers):
        # Add one in case label starts with 1. No impact if label starts with 0.
        num_classes = 10 + 1

        layers, pad, block_func = self._get_layers(num_layers)
        tf.logging.info('ResNet %d layers', num_layers if num_layers > 0 else 18)

        input_data_format = self._detect_data_format(x)
        x = self._transform_data_format(x, input_data_format, self._data_format)

        # Image standardization.
        x = x / 128 - 1.0

        # stage 1
        with tf.name_scope('stage') as name_scope:
            x = self._conv_bn(x, 3, 16, 1, padding='same')
            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())

        # C = 16, 4d
        C = 8
        filters = C * 4

        # rest 4 stage
        for i in range(4):
            with tf.name_scope('stage') as name_scope:
                x = block_func(x, filters, 2, C=C, pad=pad)
                for _ in range(layers[i]):
                    x = block_func(x, filters, 1, C=C)
                tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            filters *= 2

        x = self._global_avg_pool(x)
        x = self._fully_connected(x, num_classes)

        return x

    def _build_block(self, x, filters, strides=1, C=32, pad=None):
        """ResNeXt block

        Arguments:
            x {Tensor} -- input
            filters {int} -- number of filters
            strides {int} -- strides
            C {int} -- cardinatlity
            pad {any} -- pad
        """
        # if self._data_format == 'channels_first':
        #     in_filters = x.shape[1]
        # else:
        #     in_filters = x.shape[3]

        with tf.name_scope('build_block') as name_scope:
            orig_x = x
            layers = []

            padding = 'valid' if strides > 1 else 'same'
            for _ in range(C):
                x = orig_x
                x = self._conv_bn(x, 1, filters // C, strides, pad, padding=padding)
                x = self._conv_bn(x, 3, filters // C, padding='same')
                layers.append(x)

            if self._data_format == 'channels_first':
                x = tf.concat(layers, 1)
            else:
                x = tf.concat(layers, 3)

            x = self._conv_bn(x, 1, filters * 2, padding='same')

            if padding == 'valid':
                orig_x = self._conv(orig_x, 1, filters * 2, strides, pad, padding=padding)

            x = tf.nn.relu(tf.add(x, orig_x))

            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            return x


def build_model(input_layer, is_training, args, **kwargs):
    resnext = ResNeXt(is_training,
                    args.data_format,
                    args.batch_norm_decay,
                    args.batch_norm_epsilon)
    return resnext.build_model(input_layer, args.num_layers)
