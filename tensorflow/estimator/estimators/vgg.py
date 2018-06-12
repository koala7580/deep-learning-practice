# -*- coding: utf-8 -*-
"""VGG model

论文地址：https://arxiv.org/pdf/1409.1556.pdf
参考文章：https://hackernoon.com/learning-keras-by-implementing-vgg16-from-scratch-d036733f2d5

结果记录：
2018-06-10 step=100000 loss = 0.73 accuracy=0.7412
"""
import tensorflow as tf
from estimators.utils import build_model_fn


def conv2d_layer(inputs, filters, **kwargs):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu, **kwargs)


def max_pool_layer(inputs, **kwargs):
        return tf.layers.max_pooling2d(inputs, 2, 2, **kwargs)


def construct_model(input_layer, is_training):
    """Construct the model."""
    # Group 1
    conv1_1 = conv2d_layer(input_layer, 64, name='conv1_1')
    conv1_2 = conv2d_layer(conv1_1, 64, name='conv1_2')
    pool1 = max_pool_layer(conv1_2)

    # Group 2
    conv2_1 = conv2d_layer(pool1, 128, name='conv2_1')
    conv2_2 = conv2d_layer(conv2_1, 128, name='conv2_2')
    pool2 = max_pool_layer(conv2_2)

    # Group 3
    conv3_1 = conv2d_layer(pool2, 256, name='conv3_1')
    conv3_2 = conv2d_layer(conv3_1, 256, name='conv3_2')
    conv3_3 = conv2d_layer(conv3_2, 256, name='conv3_3')
    pool3 = max_pool_layer(conv3_3)

    # Group 4
    conv4_1 = conv2d_layer(pool3, 512, name='conv4_1')
    conv4_2 = conv2d_layer(conv4_1, 512, name='conv4_2')
    conv4_3 = conv2d_layer(conv4_2, 512, name='conv4_3')
    pool4 = max_pool_layer(conv4_3)

    # Group 5
    conv5_1 = conv2d_layer(pool4, 512, name='conv5_1')
    conv5_2 = conv2d_layer(conv5_1, 512, name='conv5_2')
    conv5_3 = conv2d_layer(conv5_2, 512, name='conv5_3')
    pool5 = max_pool_layer(conv5_3)

    # Dense Layer
    flatten = tf.layers.flatten(pool5)
    dense1 = tf.layers.dense(flatten, 4096, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, 4096, activation=tf.nn.relu)
    dense3 = tf.layers.dense(dense2, 4096, activation=tf.nn.relu)
    dropout = tf.layers.dropout(dense3, rate=0.4, training=is_training)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits


def build_estimator(config, params):
    """Build an estimator for train and predict.
    """
    return tf.estimator.Estimator(
        model_fn=build_model_fn(construct_model, resize_image=True),
        config=config,
        params=params
    )
