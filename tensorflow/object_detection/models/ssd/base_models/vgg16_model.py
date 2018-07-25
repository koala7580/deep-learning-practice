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
"""Contains definitions for VGG-16 Networks.

VGG networks were originally proposed in:
[1] Karen simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    arXiv:1409.1556
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

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
# VGG model definitions.
################################################################################
class Model(object):
    """Base class for building the Resnet Model."""

    def __init__(self, num_classes, num_filters,
                 data_format=None,
                 dtype=DEFAULT_DTYPE):
        """Creates a model for classifying an image.

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
        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.dtype = dtype
        self.training = False
        self.layers = dict()

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                             *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary.

        This function is a custom getter. A custom getter is a function with the
        same signature as tf.get_variable, except it has an additional getter
        parameter. Custom getters can be passed as the `custom_getter` parameter of
        tf.variable_scope. Then, tf.get_variable will call the custom getter,
        instead of directly getting a variable itself. This can be used to change
        the types of variables that are retrieved with tf.get_variable.
        The `getter` parameter is the underlying variable getter, that would have
        been called if no custom getter was used. Custom getters typically get a
        variable with `getter`, then modify it in some way.

        This custom getter will create an fp32 variable. If a low precision
        (e.g. float16) variable was requested it will then cast the variable to the
        requested dtype. The reason we do not directly create variables in low
        precision dtypes is that applying small gradients to such variables may
        cause the variable not to change.

        Args:
          getter: The underlying variable getter, that has the same signature as
            tf.get_variable and returns a variable.
          name: The name of the variable to get.
          shape: The shape of the variable to get.
          dtype: The dtype of the variable to get. Note that if this is a low
            precision dtype, the variable will be created as a tf.float32 variable,
            then cast to the appropriate dtype
          *args: Additional arguments to pass unmodified to getter.
          **kwargs: Additional keyword arguments to pass unmodified to getter.

        Returns:
          A variable which is cast to fp16 if necessary.
        """

        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self, name):
        """Returns a variable scope that the model should be created under.

        If self.dtype is a castable type, model variable will be created in fp32
        then cast to self.dtype before being used.

        Returns:
          A variable scope for the model.
        """

        return tf.variable_scope(name,
                                 custom_getter=self._custom_dtype_getter)

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
        with self._model_variable_scope('vgg16_model'):
            net = self._building_block(inputs,
                                       self.num_filters, # 64
                                       block_index=1,
                                       num_layers=2)

            net = self._building_block(net,
                                       self.num_filters * 2,  # 128
                                       block_index=2,
                                       num_layers=2)

            net = self._building_block(net,
                                       self.num_filters * 4,  # 256
                                       block_index=3,
                                       num_layers=3)

            net = self._building_block(net,
                                       self.num_filters * 8,  # 512
                                       block_index=4,
                                       num_layers=3)

            net = self._building_block(net,
                                       self.num_filters * 8,  # 512
                                       block_index=5,
                                       num_layers=3)

        return net

    def _building_block(self, inputs, filters, num_layers, block_index):
        """A single block for VGG-16.

          Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
              [batch, height_in, width_in, channels] depending on data_format.
            filters: The number of filters for the convolutions.
            num_layers: The number of layers in the block.
            block_index: The index of this block.

          Returns:
              The output tensor and layers collector.
        """
        net = None
        for n in range(1, num_layers + 1):
            net = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3,
                strides=1, data_format=self.data_format)
            net = batch_norm(net, self.training, self.data_format)
            net = self._activation(net)

            name = 'conv%d_%d' % (block_index, n)
            self.layers[name] = net

        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')

        name = 'pool%d' % block_index
        self.layers[name] = net

        return net

    @staticmethod
    def _activation(x):
        return tf.nn.elu(x)
