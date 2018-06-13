# -*- coding: utf-8 -*-
"""ResNet

论文地址：https://arxiv.org/pdf/1512.03385.pdf
参考文章：https://blog.csdn.net/qq_25491201/article/details/78405549

结果记录：
ResNet-18
    2018-06-12 step=100000 loss = 0.5858 accuracy=0.8402
"""
import tensorflow as tf
from estimators.utils import build_model_fn

NUM_LAYERS = 18

def conv2d_bn(inputs, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = tf.layers.conv2d(inputs, filters, kernel_size,
                         strides=strides,
                         padding=padding,
                         activation=tf.nn.relu,
                         name=conv_name)
    return tf.layers.batch_normalization(x, 3, name=bn_name)


def identity_block(inputs, filters, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = conv2d_bn(inputs, filters, kernel_size, strides, padding='same')
    x = conv2d_bn(x, filters, kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = conv2d_bn(inputs, filters, kernel_size, strides)
        return tf.add(x, shortcut)
    else:
        return tf.add(x, inputs)


def bottleneck_block(inputs, filters, strides=(1,1), with_conv_shortcut=False):
    k1, k2, k3 = filters
    x = conv2d_bn(inputs, filters=k1, kernel_size=1, strides=strides, padding='same')
    x = conv2d_bn(x, filters=k2, kernel_size=3, padding='same')
    x = conv2d_bn(x, filters=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = conv2d_bn(inputs, filters=k3, strides=strides, kernel_size=1)
        return tf.add(x, shortcut)
    else:
        return tf.add(x, inputs)


def construct_model(input_layer, is_training, **kwargs):
    """Construct the model."""
    paddings_3 = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])

    net = conv2d_bn(tf.pad(input_layer, paddings_3), 64, 7, 2, padding='valid', name='conv1')
    net = tf.layers.max_pooling2d(net, 3, 2, padding='same')

    # conv2_x
    if NUM_LAYERS < 50:
        net = identity_block(net, 64, 3)
        for _ in range(2 if NUM_LAYERS == 34 else 1):
            net = identity_block(net, 64, 3)
    else:
        net = bottleneck_block(net, (64, 64, 256), strides=1, with_conv_shortcut=True)
        net = bottleneck_block(net, (64, 64, 256))
        net = bottleneck_block(net, (64, 64, 256))

    # conv3_x
    if NUM_LAYERS < 50:
        net = identity_block(net, 128, 3, strides=2, with_conv_shortcut=True)
        for _ in range(3 if NUM_LAYERS == 34 else 1):
            net = identity_block(net, 128, 3)
    else:
        net = bottleneck_block(net, (128, 128, 512), strides=2, with_conv_shortcut=True)
        n_layers = { 50: 4, 101: 4, 152: 8 }
        for _ in range(n_layers[NUM_LAYERS] - 1):
            net = bottleneck_block(net, (128, 128, 512))

    # conv4_x
    if NUM_LAYERS < 50:
        net = identity_block(net, 256, 3, strides=2, with_conv_shortcut=True)
        for _ in range(5 if NUM_LAYERS == 34 else 1):
            net = identity_block(net, 256, 3)
    else:
        net = bottleneck_block(net, (256, 256, 1024), strides=2, with_conv_shortcut=True)
        n_layers = { 50: 6, 101: 23, 152: 36 }
        for _ in range(n_layers[NUM_LAYERS] - 1):
            net = bottleneck_block(net, (256, 256, 1024))

    # conv5_x
    if NUM_LAYERS < 50:
        net = identity_block(net, 512, 3, strides=2, with_conv_shortcut=True)
        net = identity_block(net, 512, 3)
        if NUM_LAYERS == 34:
            net = identity_block(net, 512, 3)
    else:
        net = bottleneck_block(net, (512, 512, 2048), strides=2, with_conv_shortcut=True)
        net = bottleneck_block(net, (512, 512, 2048))
        net = bottleneck_block(net, (512, 512, 2048))

    net = tf.layers.average_pooling2d(net, 7, 1)

    # Dense Layer
    flatten = tf.layers.flatten(net)

    # Logits Layer
    logits = tf.layers.dense(flatten, units=10)

    return logits


def build_estimator(config, params):
    """Build an estimator for train and predict.
    """
    return tf.estimator.Estimator(
        model_fn=build_model_fn(construct_model, resize_image=True),
        config=config,
        params=params
    )
