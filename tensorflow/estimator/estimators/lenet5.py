# -*- coding: utf-8 -*-
"""LeNet-5

中文教程：https://blog.csdn.net/d5224/article/details/68928083
论文地址：http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

结果记录：
2018-06-09 step=100000 loss = 1.16 accuracy=0.5826
"""
import os

import numpy as np
import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Estimator model function.

    Arguments:
        features {Tensor} -- features in input
        labels {Tensor} -- labels of features
        mode {tf.estimator.ModeKeys} -- mode key
        params {any} -- model params
    """
    # Use `input_layer` to apply the feature columns.
    input_layer = tf.reshape(features['image'], [-1, 32, 32, 3])

    logits = construct_model(input_layer, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    export_outputs = {
        'predict_output': tf.estimator.export.PredictOutput(predictions)
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    accuracy_mean = tf.reduce_mean(accuracy, name='accuracy_mean')
    tf.summary.scalar('accuracy', accuracy_mean)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = { "accuracy": accuracy }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


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
        model_fn=model_fn,
        config=config,
        params=params
    )
