# -*- coding: utf-8 -*-
"""Vgg 模型"""
import os

import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


class VggEstimator(tf.estimator.Estimator):
    """CIFAR-10 Vgg Estimator."""

    def __init__(self, params=None, config=None):
        """Init the estimator.
        """
        super().__init__(
            model_fn=self.the_model_fn,
            model_dir=FLAGS.model_dir,
            config=config,
            params=params
        )


    def the_model_fn(self, features, labels, mode, params):
        """Estimator model function.

        Arguments:
            features {Tensor} -- features in input
            labels {Tensor} -- labels of features
            mode {tf.estimator.ModeKeys} -- mode key
            params {any} -- model params
        """
        # Use `input_layer` to apply the feature columns.
        input_images = tf.reshape(features, [-1, 32, 32, 3])
        input_layer = tf.image.resize_images(input_images, [224, 224])

        logits = self.construct_model_vgg16(input_layer, mode == tf.estimator.ModeKeys.TRAIN)

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


    def conv_layer(self, inputs, filters):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu)


    def pool_layer(self, inputs):
        return tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2)


    def construct_model_vgg16(self, input_layer, is_training):
        """Construct the model."""
        # Convolutional Layer group 1
        conv1_1 = self.conv_layer(input_layer, 64)
        conv1_2 = self.conv_layer(conv1_1, 64)
        pool1 = self.pool_layer(conv1_2)

        # Convolutional Layer group 2
        conv2_1 = self.conv_layer(pool1, 128)
        conv2_2 = self.conv_layer(conv2_1, 128)
        pool2 = self.pool_layer(conv2_2)

        # Convolutional Layer group 3
        conv3_1 = self.conv_layer(pool2, 256)
        conv3_2 = self.conv_layer(conv3_1, 256)
        conv3_3 = self.conv_layer(conv3_2, 256)
        pool3 = self.pool_layer(conv3_3)

        # Convolutional Layer group 4
        conv4_1 = self.conv_layer(pool3, 512)
        conv4_2 = self.conv_layer(conv4_1, 512)
        conv4_3 = self.conv_layer(conv4_2, 512)
        pool4 = self.pool_layer(conv4_3)

        # Convolutional Layer group 5
        conv5_1 = self.conv_layer(pool4, 512)
        conv5_2 = self.conv_layer(conv5_1, 512)
        conv5_3 = self.conv_layer(conv5_2, 512)
        pool5 = self.pool_layer(conv5_3)

        # Dense Layer
        pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
        fc1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=4096, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=fc2, rate=0.4, training=is_training)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        return logits


if __name__ == '__main__':
    print('Not runable')
