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
"""Pascal VOC data set.
"""
import numpy as np
import tensorflow as tf

from . import base


class PascalVOCDataSet(base.DataSet):
    """Kline data set.
    """
    CLASS_NAMES = [
        'aeroplane',
        'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog',
        'horse',
        'motorbike',
        'person', 'pottedplant',
        'sheep', 'sofa',
        'train', 'tvmonitor',
    ]
    DATA_SET_NAME = 'pascal_voc'

    def __init__(self, data_dir, subset):
        super(PascalVOCDataSet, self).__init__(data_dir, subset, True)

        self.feature_dict = {
            'image': tf.FixedLenFeature([], tf.string),
            'image/shape': tf.FixedLenFeature([], tf.int64),
            'objects': tf.FixedLenFeature([], tf.string),
            'bbox/xmin': tf.FixedLenFeature([], tf.int64),
            'bbox/xmax': tf.FixedLenFeature([], tf.int64),
            'bbox/ymin': tf.FixedLenFeature([], tf.int64),
            'bbox/ymax': tf.FixedLenFeature([], tf.int64),
        }

    def extract_label(self, features):
        objects = [1 if label in self.CLASS_NAMES else 0
                   for label in features['objects']]

        bbox = []
        for i in range(len(objects)):
            xmin = features['bbox/xmin'][i]
            xmax = features['bbox/xmax'][i]
            ymin = features['bbox/ymin'][i]
            ymax = features['bbox/ymax'][i]
            bbox.append([xmin, xmax, ymin, ymax])

        return np.array(objects), np.array(bbox)

    def extract_image(self, features):
        shape = features['image_shape']
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([shape[0] * shape[1] * shape[2]])

        image = tf.reshape(image, shape)
        image = tf.cast(image, tf.float32)

        return tf.image.per_image_standardization(image)

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 20000
        elif subset == 'validation':
            return 200
        elif subset == 'test':
            return 0
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


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
