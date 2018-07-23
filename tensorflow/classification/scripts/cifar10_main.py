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
"""Runs a model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from utils.flags import core as flags_core
from utils.logs import logger
from utils import run_loop
from models.resnet_model import Cifar10ResNetModel
from datasets.cifar10 import Cifar10DataSet

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
_NUM_CLASSES = 10

_NUM_IMAGES = {
    'train': 40000,
    'validation': 10000,
    'test': 10000,
}

DATASET_NAME = 'CIFAR-10'


###############################################################################
# Data processing
###############################################################################
def get_synth_input_fn():
    return run_loop.get_synth_input_fn(
        _HEIGHT, _WIDTH, _NUM_CHANNELS, _NUM_CLASSES)


###############################################################################
# Running the model
###############################################################################
def cifar10_model_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

    learning_rate_fn = run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=128,
        num_images=_NUM_IMAGES['train'], boundary_epochs=[100, 150, 200],
        decay_rates=[1, 0.1, 0.01, 0.001])

    # We use a weight decay of 0.0002, which performs better
    # than the 0.0001 that was originally suggested.
    weight_decay = 2e-4

    # Empirical testing showed that including batch_normalization variables
    # in the calculation of regularized loss helped validation accuracy
    # for the CIFAR-10 dataset, perhaps because the regularization prevents
    # overfitting on the small data set. We therefore include all vars when
    # regularizing and computing loss during training.
    def loss_filter_fn(_):
        return True

    model = Cifar10ResNetModel(resnet_size=32,
                               resnet_version=2,
                               data_format=params['data_format'],
                               dtype=params['dtype'])

    return run_loop.model_fn(
        features=features,
        labels=labels,
        mode=mode,
        model=model,
        weight_decay=weight_decay,
        learning_rate_fn=learning_rate_fn,
        momentum=0.9,
        loss_scale=params['loss_scale'],
        loss_filter_fn=loss_filter_fn,
        dtype=params['dtype']
    )


def define_cifar_flags():
    run_loop.define_resnet_flags()
    flags.adopt_module_key_flags(run_loop)
    flags_core.set_defaults(data_dir=os.environ.get('TF_DATA_DIR', '/tmp/cifar10_data'),
                            model_dir=os.environ.get('TF_MODEL_DIR', '/tmp/cifar10_model'),
                            resnet_size='32',
                            train_epochs=250,
                            epochs_between_evals=10,
                            batch_size=128)


def run_cifar(flags_obj):
    """Run CIFAR-10 training and eval loop.

    Args:
      flags_obj: An object containing parsed flag values.
    """
    input_function = (flags_obj.use_synthetic_data and get_synth_input_fn()
                      or Cifar10DataSet.input_fn)
    run_loop.main(
        flags_obj, 'resnet', cifar10_model_fn, input_function, DATASET_NAME,
        shape=[_HEIGHT, _WIDTH, _NUM_CHANNELS])


def main(_):
    with logger.benchmark_context(flags.FLAGS):
        run_cifar(flags.FLAGS)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_cifar_flags()
    absl_app.run(main)
