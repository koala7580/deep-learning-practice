# -*- coding: utf-8 -*-
"""LeNet-5

中文教程：https://blog.csdn.net/d5224/article/details/68928083
论文地址：http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

结果记录：
2018-06-09 step=100000 loss = 1.16 accuracy=0.5826
"""
import tensorflow as tf
from estimators.utils import build_model_fn


def construct_model(input_layer, is_training):
    """Construct the model."""
    # C1
    conv_c1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=6,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu,
        name="C1")

    # S2
    pool_s2 = tf.layers.max_pooling2d(inputs=conv_c1, pool_size=[2, 2], strides=2, name="S2")

    # C3
    conv_c3 = tf.layers.conv2d(
        inputs=pool_s2,
        filters=16,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu,
        name="C3")
    # S4
    pool_s4 = tf.layers.max_pooling2d(inputs=conv_c3, pool_size=[2, 2], strides=2, name="S4")

    # C5
    conv_c5 = tf.layers.conv2d(
        inputs=pool_s4,
        filters=120,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu,
        name="C5")

    # Dense Layer
    flatten = tf.reshape(conv_c5, [-1, 120])
    dense = tf.layers.dense(inputs=flatten, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=is_training)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits


def build_estimator(config, params):
    """Build an estimator for train and predict.
    """
    return tf.estimator.Estimator(
        model_fn=build_model_fn(construct_model, resize_image=False),
        config=config,
        params=params
    )
