# -*- coding: utf-8 -*-
"""ResNet

论文地址：https://arxiv.org/pdf/1512.03385.pdf
参考文章：https://blog.csdn.net/qq_25491201/article/details/78405549

结果记录：
ResNet-18
    2018-06-12 step=100000 loss = 0.5858 accuracy=0.8402
"""
import tensorflow as tf
from estimators.base import BaseEstimator

class ResNet(BaseEstimator):
    """ResNet model."""

    def __init__(self, is_training, data_format, batch_norm_decay, batch_norm_epsilon):
        """ResNet constructor.
        Args:
        is_training: if build training or inference model.
        data_format: the data_format used during computation.
                    one of 'channels_first' or 'channels_last'.
        """
        super(ResNet, self).__init__(
            is_training,
            data_format,
            batch_norm_decay,
            batch_norm_epsilon)

    def _get_layers(self, num_layers):
        if num_layers == 18:
            layers = [1, 1, 1, 1]
            pad = [0, 1, 0, 1]
            res_func = self._identity_block
        elif num_layers == 34:
            layers = [2, 3, 5, 2]
            pad = [0, 1, 0, 1]
            res_func = self._identity_block
        elif num_layers == 50:
            layers = [2, 3, 5, 2]
            pad = None
            res_func = self._bottleneck_block
        else:
            raise ValueError('%d layers is not supportted', num_layers)

        return layers, pad, res_func

    def build_model(self, x, num_layers):
        # Add one in case label starts with 1. No impact if label starts with 0.
        num_classes = 10 + 1
        filters = [16, 16, 32, 64]

        layers, pad, res_func = self._get_layers(num_layers)
        tf.logging.info('ResNet %d layers', num_layers if num_layers > 0 else 18)

        input_data_format = self._detect_data_format(x)
        x = self._transform_data_format(x, input_data_format, self._data_format)

        # Image standardization.
        x = x / 128 - 1.0

        # stage 1
        with tf.name_scope('stage') as name_scope:
            x = self._conv_bn(x, 3, 16, 1, padding='same')
            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())

        # rest 4 stage
        for i in range(4):
            with tf.name_scope('stage') as name_scope:
                x = res_func(x, filters[i], 2, pad=pad)
                for _ in range(layers[i]):
                    x = res_func(x, filters[i], 1)
                tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())

        x = self._global_avg_pool(x)
        x = self._fully_connected(x, num_classes)

        return x

    def _identity_block(self, x, filters, strides=1, pad=None):
        """Identity residual block

        Arguments:
            x {Tensor} -- input
            in_filters {int} -- number of input filters
            filters {int} -- number of filters
            strides {int} -- strides
            pad {any} -- pad
        """
        if self._data_format == 'channels_first':
            in_filters = x.shape[1]
        else:
            in_filters = x.shape[3]

        with tf.name_scope('identity_block') as name_scope:
            orig_x = x

            padding = 'valid' if strides > 1 else 'same'
            x = self._conv_bn(x, 3, filters, strides, pad, padding=padding)

            x = self._conv_bn(x, 3, filters, padding='same', activation=None)

            if in_filters != filters or strides > 1:
                orig_x = self._conv(orig_x, 1, filters, strides, padding='same')

            x = tf.nn.relu(tf.add(x, orig_x))

            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            return x

    def _bottleneck_block(self, x, filters, strides=1, pad=None):
        """Bottleneck residual block

        Arguments:
            x {Tensor} -- input
            in_filters {int} -- number of input filters
            filters {int} -- number of filters
            strides {int} -- strides
            pad {any} -- pad
        """
        if self._data_format == 'channels_first':
            in_filters = x.shape[1]
        else:
            in_filters = x.shape[3]

        with tf.name_scope('bottle_residual_v2') as name_scope:
            orig_x = x

            padding = 'valid' if strides > 1 else 'same'
            x = self._conv_bn(x, 1, filters // 4, strides, pad, padding=padding)
            x = self._conv_bn(x, 3, filters // 4, 1, padding='same')

            x = self._conv_bn(x, 1, filters // 1, 1, padding='same')

            if in_filters != filters or strides > 1:
                orig_x = self._conv(orig_x, 1, filters, strides, pad, padding=padding)
            x = tf.add(x, orig_x)

            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            return x


def build_model(input_layer, is_training, args, **kwargs):
    resnet = ResNet(is_training,
                    args.data_format,
                    args.batch_norm_decay,
                    args.batch_norm_epsilon)
    return resnet.build_model(input_layer, args.num_layers)
