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
"""K-line data set.
"""
import numpy as np
import tensorflow as tf

from . import base


class KlineDataSet(base.DataSet):
    """Kline data set.
    """

    HEIGHT = 224
    WIDTH = 224 * 3
    DEPTH = 3

    DATA_SET_NAME = 'kline'

    def __init__(self, data_dir, subset):
        super(KlineDataSet, self).__init__(data_dir, subset, True)
        self.compression_type = 'ZLIB'
        self.shuffle_factor = 0.1

        self.feature_dict = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'buy_date': tf.FixedLenFeature([], tf.string),
            'sell_date': tf.FixedLenFeature([], tf.string),
            'code': tf.FixedLenFeature([], tf.string),
        }

    def extract_label(self, features):
        return tf.cast(features['label'], tf.int32)

    def extract_image(self, features):
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([self.HEIGHT * self.WIDTH * self.DEPTH])

        image = tf.cast(tf.reshape(image, [self.HEIGHT, self.WIDTH, self.DEPTH]),
                        tf.float32)

        if self.subset == 'train':
            image = tf.image.random_hue(image, 0.25)
            # image = tf.image.random_saturation(image)
            # image = tf.image.random_brightness(image)

            noise = tf.random_uniform(image.shape, -32, 32)
            image = image + noise

        image = tf.image.per_image_standardization(image)

        return image

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 33679
        elif subset == 'validation':
            return 736
        elif subset == 'test':
            return 0
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

    @staticmethod
    def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
        """Input_fn using the tf.data input pipeline for K-line dataset.

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
            dataset = KlineDataSet(data_dir, subset)
            return dataset.make_batch(batch_size, num_epochs, num_gpus)
