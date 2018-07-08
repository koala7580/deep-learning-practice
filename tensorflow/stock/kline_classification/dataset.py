"""data set.
"""
import os

import tensorflow as tf

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224 * 3
IMAGE_DEPTH = 3

class DataSet:
    """data set.
    """

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
        filename = os.path.join(self._data_dir, 'stock_' + subset + '.tfrecords')

        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset([filename], compression_type='GZIP')

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
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'buy_date': tf.FixedLenFeature([], tf.string),
                'sell_date': tf.FixedLenFeature([], tf.string),
                'code': tf.FixedLenFeature([], tf.string),
            }
        )
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        # image = tf.transpose(tf.reshape(image, [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH]), [1, 2, 0])
        image = tf.cast(tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]), tf.float32)
        image = tf.cast(image, tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # Custom preprocessing.
        image = self._preprocess(image)

        # normalize the image to [-1, 1]
        image = tf.image.per_image_standardization(image)

        return image, label

    def _preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self._use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_HEIGHT + 4, IMAGE_WIDTH + 4)
            image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
        return image
