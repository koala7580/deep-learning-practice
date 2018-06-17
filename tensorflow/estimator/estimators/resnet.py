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

    def build_model(self, x, num_layers):
        n = (num_layers - 2) // 6
        # Add one in case label starts with 1. No impact if label starts with 0.
        num_classes = 10 + 1
        filters = [16, 16, 32, 64]
        strides = [1, 2, 2]

        if x.shape[1] == 3 or x.shape[1] == 4:
            input_data_format = 'channels_first'
        else:
            input_data_format = 'channels_last'

        x = self._transform_data_format(x, input_data_format, self._data_format)

        # Image standardization.
        x = x / 128 - 1.0

        x = self._conv(x, 3, 16, 1, name='conv1')
        x = self._batch_norm(x, name='conv1_bn')
        x = tf.nn.relu(x)

        # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
        res_func = self._residual_v1

        # 3 stages of block stacking.
        for i in range(3):
            with tf.name_scope('stage'):
                for j in range(n):
                    if j == 0:
                        # First block in a stage, filters and strides may change.
                        x = res_func(x, 3, filters[i], filters[i + 1], strides[i])
                    else:
                        # Following blocks in a stage, constant filters and unit stride.
                        x = res_func(x, 3, filters[i + 1], filters[i + 1], 1)

        x = self._global_avg_pool(x)
        x = self._fully_connected(x, num_classes)

        return x

    def _residual_v1(self,
                    x,
                    kernel_size,
                    in_filter,
                    out_filter,
                    stride,
                    activate_before_residual=False):
        """Residual unit with 2 sub layers, using Plan A for shortcut connection."""

        del activate_before_residual
        with tf.name_scope('residual_v1') as name_scope:
            orig_x = x

            x = self._conv(x, kernel_size, out_filter, stride)
            x = self._batch_norm(x)
            x = tf.nn.relu(x)

            x = self._conv(x, kernel_size, out_filter, 1)
            x = self._batch_norm(x)

            if in_filter != out_filter:
                orig_x = self._avg_pool(orig_x, stride, stride)
                pad = (out_filter - in_filter) // 2
                if self._data_format == 'channels_first':
                    orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
                else:
                    orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

            x = tf.nn.relu(tf.add(x, orig_x))

            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            return x

    def _residual_v2(self,
                    x,
                    in_filter,
                    out_filter,
                    stride,
                    activate_before_residual=False):
        """Residual unit with 2 sub layers with preactivation, plan A shortcut."""

        with tf.name_scope('residual_v2') as name_scope:
            x, orig_x = self._activate_before_residual(x, activate_before_residual)

            x = self._conv(x, 3, out_filter, stride)

            x = self._batch_norm(x)
            x = tf.nn.relu(x)
            x = self._conv(x, 3, out_filter, [1, 1, 1, 1])

            if in_filter != out_filter:
                pad = (out_filter - in_filter) // 2
                orig_x = self._avg_pool(orig_x, stride, stride)
                if self._data_format == 'channels_first':
                    orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
                else:
                    orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

            x = tf.add(x, orig_x)

            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            return x

    def _bottleneck_residual_v2(self,
                                x,
                                in_filter,
                                out_filter,
                                stride,
                                activate_before_residual=False):
        """Bottleneck residual unit with 3 sub layers, plan B shortcut."""

        with tf.name_scope('bottle_residual_v2') as name_scope:
            x, orig_x = self._activate_before_residual(x, activate_before_residual)

            x = self._conv(x, 1, out_filter // 4, stride, is_atrous=True)
            x = self._batch_norm(x)
            x = tf.nn.relu(x)

            # pad when stride isn't unit
            x = self._conv(x, 3, out_filter // 4, 1, is_atrous=True)
            x = self._batch_norm(x)
            x = tf.nn.relu(x)

            x = self._conv(x, 1, out_filter, 1, is_atrous=True)

            if in_filter != out_filter:
                orig_x = self._conv(orig_x, 1, out_filter, stride, is_atrous=True)
            x = tf.add(x, orig_x)

            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            return x

    def _activate_before_residual(self, x, activate_before_residual=False):
        if activate_before_residual:
            x = self._batch_norm(x)
            x = tf.nn.relu(x)
            orig_x = x
        else:
            orig_x = x
            x = self._batch_norm(x)
            x = tf.nn.relu(x)

        return x, orig_x

def build_model(input_layer, is_training, args, **kwargs):
    resnet = ResNet(is_training,
                    args.data_format,
                    args.batch_norm_decay,
                    args.batch_norm_epsilon)
    return resnet.build_model(input_layer, args.num_layers)
