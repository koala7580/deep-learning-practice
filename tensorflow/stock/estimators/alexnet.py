# -*- coding: utf-8 -*-
"""AlexNet

论文地址：https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
参考文章：http://lanbing510.info/2017/07/18/AlexNet-Keras.html

结果记录：
2018-06-10 step=100000 loss = 0.73 accuracy=0.7412
"""
import sys
import tensorflow as tf


def conv2d_layer(inputs, filters, kernel_size, strides=1, **kwargs):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=tf.nn.relu, **kwargs
    )


def construct_model(input_layer, is_training, **kwargs):
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
    net = tf.layers.dense(inputs=flatten, units=4096, activation=tf.nn.relu, name='fc1')
    net = tf.layers.dropout(net, rate=0.4, training=is_training, name='dropout2')
    net = tf.layers.dense(net, units=4096, activation=tf.nn.relu, name='fc3')
    net = tf.layers.dropout(net, rate=0.4, training=is_training, name='dropout4')

    # Logits Layer
    logits = tf.layers.dense(inputs=net, units=10)

    return logits


def build_estimator(config, params, build_model_fn):
    """Build an estimator for train and predict.
    """
    return tf.estimator.Estimator(
        model_fn=build_model_fn(construct_model),
        config=config,
        params=params
    )
