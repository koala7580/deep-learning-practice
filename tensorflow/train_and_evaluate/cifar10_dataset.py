"""CIFAR-10 DataSet for train and evaluate.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3

class DataSet(object):
    """DataSet"""

    def __init__(self, data_dir):
        self._data_dir = data_dir
        self._use_distortion = False

    def train_input_fn(self, batch_size, use_distortion_for_training=True):
        return lambda: self._input_fn('train', batch_size, use_distortion_for_training)

    def eval_input_fn(self, batch_size):
        return lambda: self._input_fn('eval', batch_size)

    def _input_fn(self, subset, batch_size, use_distortion_for_training=False):
        with tf.device('/cpu:0'):
            self._use_distortion = subset == 'train' and use_distortion_for_training
            image_batch, label_batch = self._make_batch(subset, batch_size)
            return { 'image': image_batch }, label_batch


    def _make_batch(self, subset, batch_size):
        """Read the images and labels from 'filenames'."""
        filename = os.path.join(self._data_dir, subset + '.tfrecords')

        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset([filename])

        # Parse records.
        dataset = dataset.map(self._parser)

        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        dataset = dataset.shuffle(buffer_size=10 * batch_size)

        # Batch it up.
        dataset = dataset.batch(batch_size).repeat()
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    def _parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0])
        image = tf.cast(image, tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # Custom preprocessing.
        image = self._preprocess(image)

        image = tf.image.per_image_standardization(image)

        return image, label

    def _preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self._use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 4, WIDTH + 4)
            image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
            image = tf.image.random_flip_left_right(image)
        return image
