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
"""CIFAR-10 data set.
See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import numpy as np
import tensorflow as tf
from . import base


class Cifar10DataSet(base.DataSet):
    """Cifar10 data set.
    Described by http://www.cs.toronto.edu/~kriz/cifar.html.
    """
    HEIGHT = 32
    WIDTH = 32
    DEPTH = 3
    DATA_SET_NAME = 'cifar10'

    def __init__(self, data_dir, subset='train', use_distortion=True):
        super(Cifar10DataSet, self).__init__(data_dir, subset, use_distortion)

        self.feature_dict = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        }

    def extract_label(self, features):
        return tf.cast(features['label'], tf.int32)

    def extract_image(self, features):
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([self.DEPTH * self.HEIGHT * self.WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.reshape(image, [self.DEPTH, self.HEIGHT, self.WIDTH])
        image = tf.transpose(image, [1, 2, 0])
        image = tf.cast(image, tf.float32)

        image = self.preprocess(image)

        return image

    def preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self.subset == 'train' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           self.HEIGHT + 8,
                                                           self.WIDTH + 8)
            image = tf.random_crop(image, [self.HEIGHT, self.WIDTH, self.DEPTH])
            image = tf.image.random_flip_left_right(image)
            noise = tf.random_uniform(image.shape, 0, 128, seed=1)
            image = image + noise

        image = tf.image.per_image_standardization(image)

        return image

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

    @staticmethod
    def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
        """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

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
            dataset = Cifar10DataSet(data_dir, subset, is_training)
            return dataset.make_batch(batch_size, num_epochs, num_gpus)
