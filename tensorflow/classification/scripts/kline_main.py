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
"""Runs a model on the K-line dataset."""

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
from models.resnet_model import KlineResNetModel
from datasets.kline import KlineDataSet

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 2

DATASET_NAME = 'K-line'


###############################################################################
# Data processing
###############################################################################
def get_synth_input_fn():
    return run_loop.get_synth_input_fn(
        _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE * 3, _NUM_CHANNELS, _NUM_CLASSES)


###############################################################################
# Running the model
###############################################################################
def kline_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""
    learning_rate_fn = run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=256,
        num_images=KlineDataSet.num_examples_per_epoch('train'),
        boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

    model = KlineResNetModel(resnet_size=18,
                             resnet_version=2,
                             data_format=params['data_format'],
                             dtype=params['dtype'])

    return run_loop.model_fn(
        features=features,
        labels=labels,
        mode=mode,
        model=model,
        weight_decay=1e-4,
        learning_rate_fn=learning_rate_fn,
        momentum=0.9,
        loss_scale=params['loss_scale'],
        loss_filter_fn=None,
        dtype=params['dtype']
    )


def define_kline_flags():
    run_loop.define_resnet_flags(
        resnet_size_choices=['18', '34', '50', '101', '152', '200'])
    flags.adopt_module_key_flags(run_loop)
    flags_core.set_defaults(data_dir=os.environ.get('TF_DATA_DIR', '/tmp/kline_data'),
                            model_dir=os.environ.get('TF_MODEL_DIR', '/tmp/kline_model'),
                            train_epochs=100)


def run_kline(flags_obj):
    """Run ResNet ImageNet training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
    input_function = (flags_obj.use_synthetic_data and get_synth_input_fn()
                      or KlineDataSet.input_fn)

    run_loop.main(
        flags_obj, 'resnet', kline_model_fn, input_function, DATASET_NAME,
        shape=[_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])


def main(_):
    with logger.benchmark_context(flags.FLAGS):
        run_kline(flags.FLAGS)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_kline_flags()
    absl_app.run(main)
