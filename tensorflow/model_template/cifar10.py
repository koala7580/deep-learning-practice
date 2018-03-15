# -*- coding: utf-8 -*-
"""Example model: CIFAR10
"""
import re
import os
import sys
import tarfile
import model
import cifar10_input
import tensorflow as tf
from six.moves import urllib


class CIFAR10Model(model.Model):
  """Example model."""

  def __init__(self):
    """Init model."""
    super(CIFAR10Model, self).__init__()

    # Global constants describing the CIFAR-10 data set.
    self.IMAGE_SIZE = cifar10_input.IMAGE_SIZE
    self.NUM_CLASSES = cifar10_input.NUM_CLASSES
    self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # Constants describing the training process.
    self.MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
    self.NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
    self.LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    self.INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

    # If a model is trained with multiple GPU's prefix all Op names with tower_name
    # to differentiate the operations. Note that this prefix is removed from the
    # names of the summaries when visualizing a model.
    self.TOWER_NAME = 'tower'

    self.DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

  def distorted_inputs(self):
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not self.FLAGS.data_dir:
      raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(self.FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar10_input.distorted_inputs(data_dir=data_dir,
                                          batch_size=self.FLAGS.batch_size)

  def inputs(self, eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not self.FLAGS.data_dir:
      raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(self.FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                                batch_size=self.FLAGS.batch_size)

  def inference(self, images):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
      kernel = self._variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                                stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(bias, name=scope.name)
      self._activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
      kernel = self._variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                                stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(bias, name=scope.name)
      self._activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
      # Move everything into depth so we can perform a single matrix multiply.
      dim = 1
      for d in pool2.get_shape()[1:].as_list():
        dim *= d
      reshape = tf.reshape(pool2, [self.FLAGS.batch_size, dim])

      weights = self._variable_with_weight_decay('weights', shape=[dim, 384],
                                                  stddev=0.04, wd=0.004)
      biases = self._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      self._activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
      weights = self._variable_with_weight_decay('weights', shape=[384, 192],
                                            stddev=0.04, wd=0.004)
      biases = self._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
      self._activation_summary(local4)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
      weights = self._variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                                  stddev=1/192.0, wd=0.0)
      biases = self._variable_on_cpu('biases', [NUM_CLASSES],
                                      tf.constant_initializer(0.0))
      softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
      self._activation_summary(softmax_linear)

    return softmax_linear


  def maybe_download_and_extract(self):
    """Download and extract the tarball from Alex's website."""
    dest_directory = self.FLAGS.data_dir
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = self.DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(self.DATA_URL, filepath,
                                              reporthook=_progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)
