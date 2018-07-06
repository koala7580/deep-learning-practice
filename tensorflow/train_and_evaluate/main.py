"""example for tf.estimator.train_and_evaluate
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import argparse

import tensorflow as tf

from cifar10_dataset import DataSet

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

    flatten = tf.layers.flatten(inputs)

    xavier_initializer = tf.contrib.layers.xavier_initializer()
    dense = tf.layers.dense(flatten, units=4096, activation=tf.nn.relu, kernel_initializer=xavier_initializer)
    logits = tf.layers.dense(dense, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # export_outputs = {
    #     'predict_output': tf.estimator.export.PredictOutput(predictions)
    # }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)
            # export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops={ "accuracy": accuracy })

def main(args):
    config = tf.estimator.RunConfig()
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=args.job_dir, config=config, params={})

    dataset = DataSet(args.data_dir)
    train_spec = tf.estimator.TrainSpec(dataset.train_input_fn(args.train_batch_size), max_steps=args.train_steps)
    eval_spec = tf.estimator.EvalSpec(dataset.eval_input_fn(args.eval_batch_size))

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
        default=128,
        help='Batch size for training.')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=100,
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
