#!/usr/bin/env python
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
"""Runs a CNN model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import importlib

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from app.utils import run_loop
from app.utils.logs import logger
from app.models import resnet_model
from app.datasets.cifar10 import input_fn

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

DATASET_NAME = 'CIFAR-10'


###############################################################################
# Running the model
###############################################################################
class Cifar10Model(resnet_model.Model):
    """Model class with appropriate defaults for CIFAR-10 data."""

    def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
                 resnet_version=resnet_model.DEFAULT_VERSION,
                 dtype=resnet_model.DEFAULT_DTYPE):
        """These are the parameters that work for CIFAR-10 data.

        Args:
          resnet_size: The number of convolutional layers needed in the model.
          data_format: Either 'channels_first' or 'channels_last', specifying which
            data format to use when setting up the model.
          num_classes: The number of output classes needed from the model. This
            enables users to extend the same model to their own datasets.
          resnet_version: Integer representing which version of the ResNet network
          to use. See README for details. Valid values: [1, 2]
          dtype: The TensorFlow dtype to use for calculations.

        Raises:
          ValueError: if invalid resnet_size is chosen
        """
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)

        num_blocks = (resnet_size - 2) // 6

        super(Cifar10Model, self).__init__(
            resnet_size=resnet_size,
            bottleneck=False,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            block_sizes=[num_blocks] * 3,
            block_strides=[1, 2, 2],
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype
        )


def cifar10_model_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

    learning_rate_fn = run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=128,
        num_images=_NUM_IMAGES['train'], boundary_epochs=[100, 150, 200],
        decay_rates=[1, 0.1, 0.01, 0.001])

    # We use a weight decay of 0.0002, which performs better
    # than the 0.0001 that was originally suggested.
    weight_decay = params['weight_decay']

    # Empirical testing showed that including batch_normalization variables
    # in the calculation of regularized loss helped validation accuracy
    # for the CIFAR-10 dataset, perhaps because the regularization prevents
    # overfitting on the small data set. We therefore include all vars when
    # regularizing and computing loss during training.
    def loss_filter_fn(_):
        return True

    model = Cifar10Model(
        params['resnet_size'],
        data_format=params['data_format'],
        resnet_version=params['resnet_version'],
    )

    return run_loop.model_fn(
        features=features,
        labels=labels,
        mode=mode,
        model=model,
        weight_decay=weight_decay,
        learning_rate_fn=learning_rate_fn,
        momentum=params['momentum'],
        loss_scale=params['loss_scale'],
        loss_filter_fn=loss_filter_fn,
        dtype=params['dtype']
    )


def run_cifar(run_config):
    """Run ResNet CIFAR-10 training and eval loop.

      Args:
        run_config: A namedtuple containing run config values.
    """
    # input_function = (run_config.use_synthetic_data and get_synth_input_fn()
    #                   or (lambda: input_fn(run_config.data_dir, 'train')))
    input_function = lambda :input_fn(run_config.data_dir, 'train')
    run_loop.main(
        run_config, cifar10_model_fn, input_function, DATASET_NAME,
        shape=[_HEIGHT, _WIDTH, _NUM_CHANNELS])


def main():
    config_file = flags.FLAGS.config
    m = importlib.import_module(config_file)
    run_cifar(m.run_config)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    flags.DEFINE_string('-c', '--config', 'config.py')
    absl_app.run(main)
