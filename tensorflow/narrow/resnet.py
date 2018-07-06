"""ResNet model
论文地址：https://arxiv.org/pdf/1512.03385.pdf
"""
from __future__ import division

import tensorflow as tf


def conv2d(x, filters, **kwargs):
    return tf.layers.conv2d(x, filters, 3, padding="same", activation=tf.nn.elu, **kwargs)

def block(x, filters, n_layers, name):
    for i in range(n_layers):
        x = conv2d(x, filters, name='%s/conv_%d' % (name, i + 1))
    return tf.layers.max_pooling2d(x, 2, 2, name='%s/pool' % name)

def build_model_16(inputs, args, params):
    scale_factor = 4

    x = block(inputs, 64 // scale_factor, 2, 'block1')
    x = block(inputs, 128 // scale_factor, 2, 'block2')
    x = block(inputs, 256 // scale_factor, 2, 'block3')
    x = block(inputs, 512 // scale_factor, 3, 'block4')
    x = block(inputs, 512 // scale_factor, 3, 'block5')

    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)
    x = tf.layers.dropout(x, args.dropout_rate)

    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)
    x = tf.layers.dropout(x, args.dropout_rate)

    logits = tf.layers.dense(x, 10)

    return logits
