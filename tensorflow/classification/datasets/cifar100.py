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
"""CIFAR-100 data set.
See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import tensorflow as tf
from .cifar10 import Cifar10DataSet


class Cifar100DataSet(Cifar10DataSet):
    """Cifar100 data set.
    Described by http://www.cs.toronto.edu/~kriz/cifar.html.
    """
    DATA_SET_NAME = 'cifar100'

    def __init__(self, data_dir, subset='train', use_distortion=True):
        super(Cifar100DataSet, self).__init__(data_dir, subset, use_distortion)

        self.feature_dict = {
            'image': tf.FixedLenFeature([], tf.string),
            'fine_label': tf.FixedLenFeature([], tf.int64),
            'coarse_label': tf.FixedLenFeature([], tf.int64),
        }

    def extract_label(self, features):
        return tf.cast(features['fine_label'], tf.int32)

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 45000
        elif subset == 'validation':
            return 5000
        elif subset == 'test':
            return 10000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
    """Input_fn using the tf.data input pipeline for CIFAR-100 dataset.

      Args:
        is_training: A boolean denoting whether the input is for training.
        data_dir: The directory containing the input data.
        batch_size: The number of samples per batch.
        num_epochs: The number of epochs to repeat the dataset.
        num_gpus: The number of gpus used for training.

      Returns:
        A dataset that can be used for iteration.
    """
    subset = 'train' if is_training else 'validation'

    with tf.device('/cpu:0'):
        dataset = Cifar100DataSet(data_dir, subset, is_training)
        return dataset.make_batch(batch_size, num_epochs, num_gpus)
