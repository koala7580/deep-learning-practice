"""Models for CIFAR-10 dataset.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse

import tensorflow as tf

from dataset import DataSet
from utils import build_model_fn

from resnet import build_model

tf.logging.set_verbosity(tf.logging.INFO)


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
        model_fn=build_model_fn(args, build_model),
        model_dir=args.job_dir,
        config=run_config,
        params={})

    dataset = DataSet(args.data_dir)

    train_spec = tf.estimator.TrainSpec(input_fn=dataset.train_input_fn(args.train_batch_size),
                                        max_steps=args.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=dataset.eval_input_fn(args.eval_batch_size),
                                      start_delay_secs=600,
                                      throttle_secs=600)

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
        default=1000000,
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
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.5,
        help="""\
        The ratio of dropout.""")
    args = parser.parse_args()

    main(args)
