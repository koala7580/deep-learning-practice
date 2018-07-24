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
"""Base class for TFRecordDataSet.
"""
import os

import tensorflow as tf


class DataSet(object):
    """Base class for data set.
    """
    DATA_SET_NAME = ''

    def __init__(self, data_dir, subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion
        self.compression_type = None
        self.feature_dict = None

    def extract_label(self, features):
        raise NotImplemented

    def extract_image(self, features):
        raise NotImplemented

    def get_filenames(self):
        if self.subset in ['train', 'validation', 'test']:
            return [os.path.join(self.data_dir,
                                 '{}_{}.tfrecords'.format(self.DATA_SET_NAME,
                                                          self.subset))]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        features = tf.parse_single_example(
            serialized_example,
            features=self.feature_dict
        )
        image = self.extract_image(features)
        label = self.extract_label(features)

        return image, label

    def make_batch(self, batch_size, num_epochs, num_gpus):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames, self.compression_type).repeat(num_epochs)

        # We prefetch a batch at a time, This can help smooth out the time taken to
        # load input files as we go through shuffling and processing.
        dataset = dataset.prefetch(buffer_size=batch_size)

        # Parse records.
        dataset = dataset.map(
            self.parser, num_parallel_calls=batch_size)

        # Potentially shuffle records.
        if self.subset == 'train':
            min_queue_examples = int(
                self.num_examples_per_epoch(self.subset) * 0.4)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

            if num_gpus:
                total_examples = (num_epochs *
                                  self.num_examples_per_epoch(self.subset))
                # Force the number of batches to be divisible by the number of devices.
                # This prevents some devices from receiving batches while others do not,
                # which can lead to a lockup. This case will soon be handled directly by
                # distribution strategies, at which point this .take() operation will no
                # longer be needed.
                total_batches = total_examples // batch_size // num_gpus * num_gpus
                dataset.take(total_batches * batch_size)

        # Batch it up.
        dataset = dataset.batch(batch_size)

        # Operations between the final prefetch and the get_next call to the iterator
        # will happen synchronously during run time. We prefetch here again to
        # background all of the above processing work and keep it out of the
        # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
        # allows DistributionStrategies to adjust how many batches to fetch based
        # on how many devices are present.
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

        return dataset

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        raise NotImplemented
