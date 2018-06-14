"""data set.
"""
import os

import tensorflow as tf

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224 * 3
IMAGe_DEPTH = 3

class DataSet:
    """data set.
    """

    def __init__(self, data_dir, subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion

    def get_filenames(self):
        if self.subset in ['train', 'eval']:
            return [os.path.join(self.data_dir, 'stock_%s.tfrecords' % self.subset)]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, serialized_example):
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
        image.set_shape([IMAGE_HEIGHT * IMAGE_WIDTH * IMAGe_DEPTH])

        image = tf.cast(tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGe_DEPTH]), tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # Custom preprocessing.
        image = self.preprocess(image)

        # normalize the image to [-1, 1]
        image = image / 128 - 1.0

        return image, label

    def make_batch(self, batch_size):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames)

        # Parse records.
        dataset = dataset.map(self.parser)

        # Potentially shuffle records.
        min_queue_examples = 0
        if self.subset == 'train':
            min_queue_examples = int(
                DataSet.num_examples_per_epoch(self.subset) * 0.4)

        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

        # Batch it up.
        dataset = dataset.batch(batch_size).repeat()
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    def preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self.subset == 'train' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_HEIGHT + 4, IMAGE_WIDTH + 4)
            image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGe_DEPTH])
        return image

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        epoch_dict = { 
            'train': 45000,
            'eval': 10000
        }
        if subset in epoch_dict:
            return epoch_dict[subset]
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


def input_fn(data_dir,
             subset,
             batch_size,
             use_distortion_for_training=True):
    """Create input graph for model.

    Args:
      data_dir: Directory where TFRecords representing the dataset are located.
      subset: one of 'train', 'validate' and 'eval'.
      batch_size: total batch size for training to be divided by the number of
      shards.
      use_distortion_for_training: True to use distortions.
    Returns:
      two lists of tensors for features and labels, each of num_shards length.
    """
    with tf.device('/cpu:0'):
        use_distortion = subset == 'train' and use_distortion_for_training
        dataset = DataSet(data_dir, subset, use_distortion)
        image_batch, label_batch = dataset.make_batch(batch_size)
        return { 'image': image_batch }, label_batch
