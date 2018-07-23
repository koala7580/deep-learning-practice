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

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

# from main.utils.logs import logger
# from main.utils import run_loop
from main.models.resnet_model import Cifar10ResNetModel
from main.utils.learning_rate import PiecewiseLearningRate
from main.datasets.cifar10 import input_fn
from main.defaultconfig import get_config

DATASET_NAME = 'CIFAR-10'


###############################################################################
# Data processing
###############################################################################
# def get_synth_input_fn():
#     return run_loop.get_synth_input_fn(
#           _HEIGHT, _WIDTH, _NUM_CHANNELS, _NUM_CLASSES)


###############################################################################
# Running the model
###############################################################################
def cifar10_model_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""
    learning_rate_fn = PiecewiseLearningRate(
        initial_learning_rate=0.1,
        batches_per_epoch=40000 // params['batch_size'],
        boundary_epochs=[0, 100, 150, 200],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

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

    # return run_loop.model_fn(
    #     features=features,
    #     labels=labels,
    #     mode=mode,
    #     model=model,
    #     weight_decay=weight_decay,
    #     learning_rate_fn=learning_rate_fn,
    #     momentum=0.9,
    #     loss_scale=params['loss_scale'],
    #     loss_filter_fn=loss_filter_fn,
    #     dtype=params['dtype']
    # )


def main(_):
    with logger.benchmark_context(flags.FLAGS):
        run_loop.classification_main(
            get_config(flags.FLAGS),
            cifar10_model_fn,
            input_fn, DATASET_NAME)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    absl_app.run(main)
