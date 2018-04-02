# -*- coding: utf-8 -*-
"""用教程中识别 MNIST 的 CNN 模型来处理 CIFAR-10.

教程地址：https://www.tensorflow.org/tutorials/layers
"""
import os

import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS 


class CNNEstimator(tf.estimator.Estimator):
    """CIFAR-10 CNN Estimator."""

    def __init__(self, params=None, config=None):
        """Init the estimator.
        """
        super().__init__(
            model_fn=self.a_model_fn,
            model_dir=FLAGS.model_dir,
            config=config,
            params=params
        )


    def a_model_fn(self, features, labels, mode, params):
        """Estimator model function.
        
        Arguments:
            features {Tensor} -- features in input
            labels {Tensor} -- labels of features
            mode {tf.estimator.ModeKeys} -- mode key
            params {any} -- model params
        """
        # Use `input_layer` to apply the feature columns.
        input_layer = tf.reshape(features, [-1, 32, 32, 3])

        logits = self.construct_model(input_layer, mode == tf.estimator.ModeKeys.TRAIN)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # tf.summary.scalar('loss', loss)

        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        # tf.summary.scalar('accuracy', accuracy)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = { "accuracy": accuracy }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    def construct_model(self, input_layer, is_training):
        """Construct the model."""
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=is_training)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        return logits


if __name__ == '__main__':
    print('Not runable')
