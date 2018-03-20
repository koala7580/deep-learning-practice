# -*- coding: utf-8 -*-
"""Experimental model: Vgg
"""
import os
import sys
import time
import tarfile
import basemodel
import vgg_input
import tensorflow as tf
from six.moves import urllib
from codelist import sh50


class Vgg16(basemodel.BaseModel):
    """Experimental model: Vgg"""

    # Global constants describing the CIFAR-10 data set.
    IMAGE_SIZE = 224
    NUM_CLASSES = 2
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


    def train_inputs(self):
        """Construct training input for CIFAR training using the Reader ops.

        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.

        Raises:
            ValueError: If no data_dir
        """
        if not self.FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(self.FLAGS.data_dir, 'cifar-10-batches-bin')
        return data_dir


    def evaluate_inputs(self, eval_data):
        """Construct evaluation input for CIFAR evaluation using the Reader ops.

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
        return data_dir


    def maybe_download_and_extract(self):
        """Generate k-line images."""
        dest_directory = self.FLAGS.data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self.DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % \
                    (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(self.DATA_URL, filepath,
                                               reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
