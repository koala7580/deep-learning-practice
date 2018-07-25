# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for SSD 300 Networks.

SSD networks were originally proposed in:
[1] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy
    SSD: Single Shot MultiBox Detector.
    arXiv:1512.02325
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .base_models.vgg16_model import Model as BaseModel
from .layers.priorbox import PriorBox

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

      Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
          [batch, height_in, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                     Should be a positive integer.
        data_format: The input format ('channels_last' or 'channels_first').

      Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


################################################################################
# SSD 300 model definitions.
################################################################################
class Model(BaseModel):
    """Base class for building the Resnet Model."""

    def __init__(self, num_classes, num_filters,
                 data_format=None,
                 dtype=DEFAULT_DTYPE):
        """Creates a model for detecting objects in an image.

        Args:
          num_classes: The number of classes used as labels.
          num_filters: The number of filters to use for the first block layer
            of the model. This number is then doubled for each subsequent block
            layer.
          data_format: Input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
          dtype: The TensorFlow dtype to use for calculations. If not specified
            tf.float32 is used.

        Raises:
          ValueError: if invalid version is selected.
        """
        super(Model, self).__init__(
            num_classes, num_filters,
            data_format, dtype
        )

    def __call__(self, inputs, training):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        self.training = training
        input_shape = inputs.shape
        if self.data_format == 'channels_first':
            img_size = (input_shape[2], input_shape[3])
        else:
            img_size = (input_shape[1], input_shape[2])

        with self._model_variable_scope('ssd300_model'):
            if self.data_format == 'channels_first':
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

        net = super(Model, self).__call__(inputs, training)

        with self._model_variable_scope('ssd300_model'):

        net = self._atrous_convolution_2d(net, filters=1024,
                                          kernel_size=3,
                                          atrous_rate=6, name='fc6')

        net = self._conv2d(net, filters=1024, kernel_size=1,
                           padding='same', name='fc7')

        net = self._conv2d(net, filters=256, kernel_size=1,
                           padding='same', name='conv6_1')

        net = self._conv2d(net, filters=512, kernel_size=3,
                           strides=2,
                           padding='same', name='conv6_2')

        net = self._conv2d(net, filters=128, kernel_size=1,
                           padding='same', name='conv7_1')

        net = self._conv2d(fixed_padding(net, 3, self.data_format),
                           filters=256, kernel_size=3,
                           strides=2,
                           padding='valid', name='conv7_2')

        net = self._conv2d(net, filters=128, kernel_size=1,
                           padding='same', name='conv8_1')

        net = self._conv2d(net, filters=256, kernel_size=3,
                           strides=2,
                           padding='same', name='conv8_2')

        if self.data_format == 'channels_first':
            net = tf.reduce_mean(net, [2, 3])
        else:
            net = tf.reduce_mean(net, [1, 2])
        self.layers['pool6'] = net

        # Prediction from conv4_3
        conv4_3_norm = self._normalize(net, 20, name='conv4_3_norm')
        num_priors = 3
        x = self._conv2d(conv4_3_norm, filters=num_priors * 4, kernel_size=3,
                         padding='same', name='conv4_3_norm_mbox_loc')
        self.layers['conv4_3_norm_mbox_loc_flat'] = tf.layers.flatten(x, name='conv4_3_norm_mbox_loc_flat')

        x = self._conv2d(conv4_3_norm, filters=num_priors * self.num_classes,
                         kernel_size=3, padding='same',
                         name='conv4_3_norm_mbox_conf')
        self.layers['conv4_3_norm_mbox_conf_flat'] = tf.layers.flatten(x, name='conv4_3_norm_mbox_conf_flat')

        prior_box = PriorBox(img_size, min_size=30.0, aspect_ratios=[2],
                            variances=[0.1, 0.1, 0.2, 0.2],
                            name='conv4_3_norm_mbox_priorbox')
        net['conv4_3_norm_mbox_priorbox'] = prior_box(conv4_3_norm)

        return net

    def _atrous_convolution_2d(self, inputs, filters, kernel_size,
                               atrous_rate, name):
        outputs = tf.layers.conv2d(inputs, filters, kernel_size,
                                   dilation_rate=atrous_rate, padding='same',
                                   data_format=self.data_format,
                                   name=name)
        output = self._activation(outputs)
        self.layers[name] = output
        return output

    def _conv2d(self, inputs, filters, kernel_size, name, **kwargs):
        output = tf.layers.conv2d(inputs, filters, kernel_size,
                                   name=name, **kwargs)

        output = self._activation(output)
        self.layers[name] = output
        return output

    def _normalize(self, inputs, scale, name):
        axis = 1 if self.data_format == 'channels_first' else 3
        n_channels = inputs.shape[axis]
        gamma = tf.get_variable('%s_gamma' % name, (n_channels, ), tf.float32,
                                initializer=tf.constant_initializer(scale))

        output = tf.nn.l2_normalize(inputs, axis)
        for i in range(n_channels):
            if self.data_format == 'channels_first':
                output[:, i, :, :] *= gamma[i]
            else:
                output[:, :, :, i] *= gamma[i]

        self.layers[name] = output

        return output
