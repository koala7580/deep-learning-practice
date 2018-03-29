# -*- coding: utf-8 -*-
"""Define the CIFAR-10 model.
"""
import os

import tensorflow as tf


# Global constants describing the CIFAR-10 data set.
# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24
NUM_CLASSES = 10


class CIFAR10Estimator(tf.estimator.Estimator):
    """CIFAR-10 Estimator.
    
    Arguments:
        tf {[type]} -- [description]
    """

    def __init__(self, model_dir=None, params=None, config=None):
        """Init the estimator.
        """
        super().__init__(
            model_fn=self.model_fn,
            model_dir=model_dir,
            config=config,
            params=params
        )


    def model_fn(self, features, labels, mode, params):
        """CIFAR-10 Estimator model function.
        
        Arguments:
            features {Tensor} -- features in input
            labels {Tensor} -- labels of features
            mode {tf.estimator.ModeKeys} -- mode key
            params {any} -- model params
        """
        # Use `input_layer` to apply the feature columns.
        input_layer = tf.feature_column.input_layer(features, params['feature_columns'])

        net = tf.layers.conv2d(input_layer,
            filters=64,
            kernel_size=[5, 5],
            activation=tf.nn.relu,
            padding='same',
            name='conv1')
        
        net = tf.layers.max_pooling2d(net, [3, 3], 2, padding='same', name='pool1')

        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        net = tf.layers.conv2d(net,
            filters=64,
            kernel_size=[5, 5],
            activation=tf.nn.relu,
            padding='same',
            name='conv2')

        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        
        net = tf.layers.max_pooling2d(net, [3, 3], 2, padding='same', name='pool2')
        

        reshape = tf.reshape(net, [input_layer.get_shape()[0], -1])

        net = tf.layers.dense(reshape, 384, activation=tf.nn.relu, name='fc1')
        net = tf.layers.dense(net, 192, activation=tf.nn.relu, 'fc2')
        net = tf.layers.dense(net, NUM_CLASSES, 'logits')
        





if __name__ == '__main__':
    print('Not runable')
