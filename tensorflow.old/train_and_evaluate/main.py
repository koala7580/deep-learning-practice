"""example for tf.estimator.train_and_evaluate
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse

import tensorflow as tf

from cifar10_dataset import DataSet

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None

def model_fn(features, labels, mode, params):
    """Estimator model function.

    Arguments:
        features {Tensor} -- features in input
        labels {Tensor} -- labels of features
        mode {tf.estimator.ModeKeys} -- mode key
        params {any} -- model params
    """
    inputs = features['image']
    tf.summary.image('input_image', inputs)


    x = tf.layers.flatten(inputs)

    xavier_initializer = tf.contrib.layers.xavier_initializer()

    x = tf.layers.dense(x, units=3072 * 2,
                        kernel_initializer=xavier_initializer,
                        bias_initializer=tf.zeros_initializer)
    x = x * x * x

    logits = tf.layers.dense(x, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = tf.train.exponential_decay(
                FLAGS.learning_rate,
                tf.train.get_global_step(),
                10000,
                0.1)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops={ "accuracy": accuracy })


def main(args):
    # Session configuration.
    gpu_options = tf.GPUOptions(force_gpu_compatible=True)
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=gpu_options)

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=args.job_dir)
    run_config = run_config.replace(session_config=session_config)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.job_dir,
        config=run_config,
        params={})

    dataset = DataSet(args.data_dir)

    train_spec = tf.estimator.TrainSpec(input_fn=dataset.train_input_fn(args.train_batch_size),
                                        max_steps=args.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=dataset.eval_input_fn(args.eval_batch_size), start_delay_secs=60)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='The directory where the input data is stored.')
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--train-steps',
        type=int,
        default=100000,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=32,
        help='Batch size for training.')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=32,
        help='Batch size for validation.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.""")
    args = parser.parse_args()

    FLAGS = args

    main(args)
