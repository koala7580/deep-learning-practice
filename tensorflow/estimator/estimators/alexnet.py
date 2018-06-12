# -*- coding: utf-8 -*-
"""AlexNet

论文地址：https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
参考文章：http://lanbing510.info/2017/07/18/AlexNet-Keras.html

结果记录：
2018-06-10 step=100000 loss = 0.73 accuracy=0.7412
"""
import tensorflow as tf
from estimators.utils import build_model_fn


def conv2d_layer(inputs, filters, kernel_size, strides=1, **kwargs):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=tf.nn.relu, **kwargs
    )


def construct_model(input_layer, is_training):
    """Construct the model."""
    net = conv2d_layer(input_layer, 96, 11, 4, name='conv1')
    net = tf.nn.lrn(net, 5, name="lrn1")

    net = tf.layers.max_pooling2d(net, 3, strides=2, name="pool1")

    net = conv2d_layer(net, 256, 5, padding='same', name='conv2')
    net = tf.nn.lrn(net, 5, name='lrn2')
    net = tf.layers.max_pooling2d(net, 3, strides=2, name="pool2")

    net = conv2d_layer(net, 384, 3, padding='same', name='conv3')
    net = conv2d_layer(net, 384, 3, padding='same', name='conv4')
    net = conv2d_layer(net, 256, 3, padding='same', name='conv5')

    net = tf.layers.max_pooling2d(net, 3, strides=2, name="pool5")

    # Dense Layer
    flatten = tf.layers.flatten(net)
    dense6 = tf.layers.dense(inputs=flatten, units=4096, activation=tf.nn.relu)
    dropout6 = tf.layers.dropout(inputs=dense6, rate=0.4, training=is_training)
    dense7 = tf.layers.dense(inputs=dropout6, units=4096, activation=tf.nn.relu)
    dropout7 = tf.layers.dropout(inputs=dense7, rate=0.4, training=is_training)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout7, units=10)

    return logits


def build_estimator(config, params):
    """Build an estimator for train and predict.
    """
    return tf.estimator.Estimator(
        model_fn=build_model_fn(construct_model, resize_image=True),
        config=config,
        params=params
    )
