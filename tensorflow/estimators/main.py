# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tensorflow estimators for classifying images from CIFAR-10 dataset.
Support single-host training with one or multiple devices.
"""
import argparse
import functools
import itertools
import os

#import cifar10
#import cifar10_model
#import cifar10_utils
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)



def main(args):
    """Main Function"""
    # def main(job_dir, data_dir, num_gpus, variable_strategy,
    #          use_distortion_for_training, log_device_placement, num_intra_threads,
    #          **hparams):

    # Session configuration.
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=args.log_device_placement,
        intra_op_parallelism_threads=args.num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    estimator = None
    train_spec = None
    eval_spec = None
    
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
        '--variable-strategy',
        choices=['CPU', 'GPU'],
        type=str,
        default='CPU',
        help='Where to locate variable operations')
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=0,
        help='The number of gpus used. Uses only CPU if set to 0.')
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
        default=0.1,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.""")
    parser.add_argument(
        '--use-distortion-for-training',
        type=bool,
        default=True,
        help='If doing image distortion for training.')
    parser.add_argument(
        '--sync',
        action='store_true',
        default=False,
        help='If present when running in a distributed environment will run on sync mode.')
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
        '--data-format',
        type=str,
        default=None,
        help="""\
        If not set, the data format best for the training device is used. 
        Allowed values: channels_first (NCHW) channels_last (NHWC).""")
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
    args = parser.parse_args()

    if args.num_gpus > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if args.num_gpus < 0:
        raise ValueError('Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
    if args.num_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                         '--variable-strategy=CPU.')
    if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')
    if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

    main(args)