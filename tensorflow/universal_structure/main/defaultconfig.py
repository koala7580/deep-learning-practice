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
"""Runtime configuration."""

from collections import namedtuple

DefaultConfigurations = namedtuple('DefaultConfigurations', [
    'hooks', 'train_epochs', 'epochs_between_evals',
    'stop_threshold', 'batch_size',
])

default_config = DefaultConfigurations(
    # Batch size for training and evaluation. When using
    # multiple gpus, this is the global batch size for
    # all devices. For example, if the batch size is 32
    # and there are 4 GPUs, each GPU will get 8 examples on
    # each step.
    batch_size=32,

    # A list of (case insensitive) strings to specify the names of
    # training hooks.
    #
    # LoggingTensorHook
    # ProfilerHook
    # ExamplesPerSecondHook
    # LoggingMetricHook
    #
    # See utils.logs.hooks_helper for details.
    hooks='LoggingTensorHook',

    # The number of epochs used to train.
    train_epochs=1,

    # The number of training epochs to run between
    # evaluations.
    epochs_between_evals=1,

    # If passed, training will stop at the earlier of
    # train_epochs and when the evaluation metric is
    # greater than or equal to stop_threshold.
    stop_threshold=None,
)
