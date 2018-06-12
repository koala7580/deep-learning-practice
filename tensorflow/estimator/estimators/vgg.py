# -*- coding: utf-8 -*-
"""VGG model

论文地址：https://arxiv.org/pdf/1409.1556.pdf
参考文章：https://hackernoon.com/learning-keras-by-implementing-vgg16-from-scratch-d036733f2d5

结果记录：
2018-06-10 step=100000 loss = 0.73 accuracy=0.7412
"""
import os

import numpy as np
import tensorflow as tf


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


def build_estimator(config, params):
    """Build an estimator for train and predict.
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=params
    )
