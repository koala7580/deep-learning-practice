"""Util functions for building CNN models.
"""
import tensorflow as tf

def conv2d_layer(inputs, filters, kernel_size, strides=1, **kwargs):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=tf.nn.relu, **kwargs)
