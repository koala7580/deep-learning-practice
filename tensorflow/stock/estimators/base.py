"""Base class for estimator.
"""
import tensorflow as tf


class BaseEstimator(object):
    """BaseEstimator."""

    def __init__(self, is_training, data_format, batch_norm_decay, batch_norm_epsilon):
        """ResNet constructor.
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

    def _detect_data_format(self, x):
        if x.shape[1] == 3 or x.shape[1] == 4:
            return 'channels_first'
        elif x.shape[3] == 3 or x.shape[3] == 4:
            return 'channels_last'
        else:
            raise ValueError('unknown data format with shape: %s', x.ge_shape())

    def _transform_data_format(self, x, from_data_format, to_data_format):
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

    def _conv_bn(self, x, kernel_size, filters, strides=(1,1), pad=None, activation=tf.nn.relu, **kwargs):
        x = self._conv(x, kernel_size, filters, strides, pad, **kwargs)
        x = self._batch_norm(x)
        if activation:
            x = activation(x)

        return x

    def _pad(self, x, pad):
        if type(pad) == int:
            return self._pad(x, ((pad, pad), (pad, pad)))

        if len(pad) == 4:
            return self._pad(x, ((pad[0], pad[1]), (pad[2], pad[3])))

        if len(pad) == 2:
            pad_h, pad_w = pad[0], pad[1]
            if type(pad_h) == int and type(pad_w) == int:
                return self._pad(x, ((pad_h, pad_h), (pad_w, pad_w)))

            if self._data_format == 'channels_first':
                x = tf.pad(x, [[0, 0], [0, 0], [pad_h[0], pad_h[1]], [pad_w[0], pad_w[1]]])
            else:
                x = tf.pad(x, [[0, 0], [pad_h[0], pad_h[1]], [pad_w[0], pad_w[1]], [0, 0]])

        return x

    def _conv(self, x, kernel_size, filters, strides=(1,1), pad=None, **kwargs):
        """Convolution."""
        if pad:
            x = self._pad(x, pad)

        return tf.layers.conv2d(
            inputs=x,
            kernel_size=kernel_size,
            filters=filters,
            strides=strides,
            use_bias=False,
            data_format=self._data_format, **kwargs)

    def _batch_norm(self, x):

        return tf.layers.batch_normalization(
            x,
            momentum=self._batch_norm_decay,
            center=True,
            scale=True,
            epsilon=self._batch_norm_epsilon,
            training=self._is_training,
            fused=True)

    def _fully_connected(self, x, out_dim):
        with tf.name_scope('fully_connected') as name_scope:
            x = tf.layers.dense(x, out_dim)

            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            return x

    def _max_pool(self, x, pool_size, stride):
        with tf.name_scope('max_pool') as name_scope:
            x = tf.layers.max_pooling2d(
                x, pool_size, stride, padding='SAME', data_format=self._data_format)

            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            return x

    def _avg_pool(self, x, pool_size, stride):
        with tf.name_scope('avg_pool') as name_scope:
            x = tf.layers.average_pooling2d(
                x, pool_size, stride, padding='SAME', data_format=self._data_format)

            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            return x

    def _global_avg_pool(self, x):
        with tf.name_scope('global_avg_pool') as name_scope:
            assert x.get_shape().ndims == 4
            if self._data_format == 'channels_first':
                x = tf.reduce_mean(x, [2, 3])
            else:
                x = tf.reduce_mean(x, [1, 2])

            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
            return x
