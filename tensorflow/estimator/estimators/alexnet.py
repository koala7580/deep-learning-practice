# -*- coding: utf-8 -*-
"""AlexNet

中文教程：https://blog.csdn.net/zyqdragon/article/details/72353420
论文地址：https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

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
    input_layer = tf.image.resize_images(features['image'], [224, 224])

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
    tf.summary.scalar('accuracy_mean', accuracy_mean)

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
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=11,
        padding="valid",
        strides=4,
        activation=tf.nn.relu,
        name="conv1")

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2, name="pool1")

    lrn1 = tf.nn.lrn(pool1, 5, name="lrn1")

    conv2 = tf.layers.conv2d(
        inputs=lrn1,
        filters=256,
        kernel_size=5,
        padding='valid',
        activation=tf.nn.relu,
        name="conv2")

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=3, strides=2, name="pool2")

    lrn2 = tf.nn.lrn(pool2, 5, name='lrn2')

    conv3 = tf.layers.conv2d(
        inputs=lrn2,
        filters=384,
        kernel_size=3,
        padding='valid',
        activation=tf.nn.relu,
        name="conv3")

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=3,
        padding='valid',
        activation=tf.nn.relu,
        name="conv4")

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=3,
        padding='valid',
        activation=tf.nn.relu,
        name="conv5")

    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=3, strides=2, name="pool5")

    # Dense Layer
    flatten = tf.reshape(pool5, [-1, 256])
    dense6 = tf.layers.dense(inputs=flatten, units=4096, activation=tf.nn.relu)
    dropout6 = tf.layers.dropout(inputs=dense6, rate=0.4, training=is_training)
    dense7 = tf.layers.dense(inputs=dropout6, units=4096, activation=tf.nn.relu)
    dropout7 = tf.layers.dropout(inputs=dense7, rate=0.4, training=is_training)
    dense8 = tf.layers.dense(inputs=dropout7, units=1000, activation=tf.nn.relu)
    dropout8 = tf.layers.dropout(inputs=dense8, rate=0.4, training=is_training)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout8, units=10)

    return logits


def build_estimator(config, params):
    """Build an estimator for train and predict.
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=params
    )
