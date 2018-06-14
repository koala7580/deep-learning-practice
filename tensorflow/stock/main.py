"""Tensorflow estimators for stock prediction.
"""
import argparse
import os

import numpy as np
import tensorflow as tf

from estimators.alexnet import build_estimator
from utils import build_model_fn
from dataset import input_fn

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
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=args.log_device_placement,
        intra_op_parallelism_threads=args.num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=args.job_dir)
    run_config = run_config.replace(session_config=session_config)

    estimator = build_estimator(run_config, {
        'learning_rate': args.learning_rate
    }, build_model_fn)

    train_input = lambda: input_fn(args.data_dir, 'train', args.train_batch_size)
    train_spec = tf.estimator.TrainSpec(train_input, max_steps=args.train_steps)

    eval_input = lambda: input_fn(args.data_dir, 'eval', args.eval_batch_size)
    exporter = tf.estimator.FinalExporter('stock', json_serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input,
                                      steps=100,
                                      exporters=[exporter],
                                      name='stock-eval')

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
        default=16,
        help='Batch size for training.')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=16,
        help='Batch size for validation.')
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
    args = parser.parse_args()

    main(args)
