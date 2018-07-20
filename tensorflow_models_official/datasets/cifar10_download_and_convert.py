# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Downloads and convert the python version of the CIFAR-10 dataset to TFRecord."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tarfile

from six.moves import urllib
from six.moves import cPickle as pickle
import tensorflow as tf

from utils import file_md5_check

# DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
MD5_SUM = 'c58f30108f718f92721af3b95e74349a'

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/tmp/cifar10_data',
    help='Directory to download data and extract the tarball')


def unpickle(file):
    if sys.version_info >= (3, 0):
        data_dict = pickle.load(file, encoding='bytes')
    else:
        data_dict = pickle.load(file)
    return data_dict


def write_to_tfrecords(tar_file, data_file, writer):
    tar_file_obj = tarfile.open(tar_file, 'r:gz')
    for file_info in tar_file_obj:
        if data_file in file_info.name:
            data_dict = unpickle(tar_file_obj.extractfile(file_info))
            if sys.version_info >= (3, 0):
                data = data_dict[b'data']
                labels = data_dict[b'labels']
            else:
                data = data_dict['data']
                labels = data_dict['labels']

            for image, label in zip(data, labels):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                        'label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label]))
                    }))
                writer.write(example.SerializeToString())

    print('Write %s into tfrecords' % (data_file, ))


def main(_):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    filename = DATA_URL.split('/')[-1]
    file_path = os.path.join(FLAGS.data_dir, filename)

    if not os.path.exists(file_path) or \
            not file_md5_check(file_path, MD5_SUM):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, 100.0 * count * block_size / total_size))
            sys.stdout.flush()

        file_path, _ = urllib.request.urlretrieve(DATA_URL, file_path, _progress)
        print()
        stat_info = os.stat(file_path)
        print('Successfully downloaded', filename, stat_info.st_size, 'bytes.')

    def create_writer(subset):
        return tf.python_io.TFRecordWriter(
            os.path.join(FLAGS.data_dir, subset + '.tfrecords'))

    with create_writer('train') as writer:
        for data_file in ['data_batch_%d' % i for i in range(1, 5)]:
            write_to_tfrecords(file_path, data_file, writer)

    with create_writer('validation') as writer:
        write_to_tfrecords(file_path, 'data_batch_5', writer)

    with create_writer('test') as writer:
        write_to_tfrecords(file_path, 'test_batch', writer)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
