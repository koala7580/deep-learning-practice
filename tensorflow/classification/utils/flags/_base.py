# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Flags which will be nearly universal across models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import tensorflow as tf


from ._conventions import help_wrap


def define_base(data_dir=True, model_dir=True, clean=True,
                num_gpu=True, export_dir=True):
    """Register base flags.

      Args:
        data_dir: Create a flag for specifying the input data directory.
        model_dir: Create a flag for specifying the model file directory.
        num_gpu: Create a flag to specify the number of GPUs used.
        export_dir: Create a flag to specify where a SavedModel should be exported.
        clean: Create a flag to specify whether to clean the model_dir before training.

      Returns:
        A list of flags for core.py to marks as key flags.
    """
    key_flags = []

    if data_dir:
        flags.DEFINE_string(
            name="data_dir", short_name="dd",
            default=os.environ.get('TF_DATA_DIR', '/tmp/tf_data'),
            help=help_wrap("The location of the input data."))
        key_flags.append("data_dir")

    if model_dir:
        flags.DEFINE_string(
            name="model_dir", short_name="md",
            default=os.environ.get('TF_MODEL_DIR', '/tmp/tf_model'),
            help=help_wrap("The location of the model checkpoint files."))
        key_flags.append("model_dir")

    if clean:
        flags.DEFINE_boolean(
            name="clean", default=False,
            help=help_wrap("If set, model_dir will be removed if it exists."))
        key_flags.append("clean")

    if num_gpu:
        flags.DEFINE_integer(
            name="num_gpus", short_name="ng",
            default=1 if tf.test.is_gpu_available() else 0,
            help=help_wrap(
                "How many GPUs to use with the DistributionStrategies API. The "
                "default is 1 if TensorFlow can detect a GPU, and 0 otherwise."))

    if export_dir:
        flags.DEFINE_string(
            name="export_dir", short_name="ed", default=None,
            help=help_wrap("If set, a SavedModel serialization of the model will "
                           "be exported to this directory at the end of training. "
                           "See the README for more details and relevant links.")
        )
        key_flags.append("export_dir")

    return key_flags


def get_num_gpus(flags_obj):
    """Treat num_gpus=-1 as 'use all'."""
    if flags_obj.num_gpus != -1:
        return flags_obj.num_gpus

    from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
    local_device_protos = device_lib.list_local_devices()
    # return sum([1 for d in local_device_protos if d.device_type == "GPU"])
    return len(list(filter(lambda d: d.device_type == 'GPU',
                           local_device_protos)))
