# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
from __future__ import absolute_import, division, print_function

import os
import argparse
import functools
import itertools

import numpy as np
import tensorflow as tf

from vgg import build_model
from cifar10_dataset import input_fn
from utils import build_model_fn, MetadataHook

tf.logging.set_verbosity(tf.logging.INFO)


def json_serving_input_fn():
    """Build the serving inputs."""
    inputs = {
        'image': tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
    }

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def main(args):
    """Main Function"""
    # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Session configuration.
    gpu_options = tf.GPUOptions(force_gpu_compatible=True, allow_growth=True)
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=args.log_device_placement,
        intra_op_parallelism_threads=args.num_intra_threads,
        gpu_options=gpu_options)

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=args.job_dir)
    run_config = run_config.replace(session_config=session_config)
    run_config = run_config.replace(save_summary_steps=1000)

    estimator = tf.estimator.Estimator(
        model_fn=build_model_fn(build_model, lambda features: features['image'], args),
        config=run_config,
        params={}
    )

    train_input = lambda: input_fn(args.data_dir, 'train', args.train_batch_size)
    hook = MetadataHook(1000, None, args.job_dir)
    train_spec = tf.estimator.TrainSpec(train_input, max_steps=args.train_steps, hooks=[hook])

    eval_input = lambda: input_fn(args.data_dir, 'validation', args.eval_batch_size)
    exporter = tf.estimator.FinalExporter('cifar10', json_serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input,
                                      steps=100,
                                      exporters=[exporter],
                                      name='cifar10-eval')

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
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for MomentumOptimizer.')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=2e-4,
        help='Weight decay for convolutions.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.""")
    parser.add_argument(
        '--num-intra-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for intra-op parallelism. When training on CPU
        set to 0 to have the system pick the appropriate number or alternatively
        set it to the number of physical CPU cores.\
        """)
    parser.add_argument(
        '--num-inter-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for inter-op parallelism. If set to 0, the
        system will pick an appropriate number.""")
    parser.add_argument(
        '--log-device-placement',
        action='store_true',
        default=False,
        help='Whether to log device placement.')
    parser.add_argument(
        '--batch-norm-decay',
        type=float,
        default=0.997,
        help='Decay for batch norm.')
    parser.add_argument(
        '--batch-norm-epsilon',
        type=float,
        default=1e-5,
        help='Epsilon for batch norm.')
    parser.add_argument(
        '--num-layers',
        type=int,
        default=18,
        help='The number of layers of the ResNet model.')
    parser.add_argument(
        '--data-format',
        type=str,
        default='channels_last',
        help="""\
        If not set, the data format best for the training device is used.
        Allowed values: channels_first (NCHW) channels_last (NHWC).\
        """)
    args = parser.parse_args()

    if args.data_format not in ['channels_first', 'channels_last']:
        raise ValueError('Invalid --data-format parameter.')

    main(args)
