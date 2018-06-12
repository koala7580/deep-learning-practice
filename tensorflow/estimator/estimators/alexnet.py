# -*- coding: utf-8 -*-
"""AlexNet

论文地址：https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
参考文章：http://lanbing510.info/2017/07/18/AlexNet-Keras.html

结果记录：
2018-06-10 step=100000 loss = 0.73 accuracy=0.7412
"""
import tensorflow as tf


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
