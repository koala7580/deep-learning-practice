"""Layers that are used to build CNN network.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers import xavier_initializer


class DepthwiseConv2D(base.Layer):

    def __init__(self,
               kernel_size,
               strides=1,
               depth_multiplier=1,
               rate=None,
               padding='same',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=xavier_initializer(),
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(DepthwiseConv2D, self).__init__(trainable=trainable, name=name,
                                activity_regularizer=activity_regularizer,
                                **kwargs)
        rank = 2
        self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = utils.normalize_tuple(strides, rank, 'strides')
        self.rate = rate
        self.depth_multiplier = depth_multiplier
        self.padding = utils.normalize_padding(padding)
        self.data_format = utils.normalize_data_format(data_format)
        self.dilation_rate = utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = base.InputSpec(ndim=4)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis].value
        kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier)

        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                        shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None

        self.input_spec = base.InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.data_format == 'channels_first':
            data_format = 'NCHW'
            strides = [1, 1, self.strides[0], self.strides[1]]
        else:
            data_format = 'NHWC'
            strides = [1, self.strides[0], self.strides[1], 1]
    
        outputs = tf.nn.depthwise_conv2d(inputs,
            self.kernel, strides, self.padding.upper(),
            self.rate, data_format=data_format
        )

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            [self.depth_multiplier])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.depth_multiplier] +
                                            new_space)
