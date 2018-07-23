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

"""Downloads and convert the python version of the CIFAR-100 dataset to TFRecord."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tarfile

from six.moves import urllib
from six.moves import cPickle
import tensorflow as tf

from practice.utils.check_sum import file_md5_check

# DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
MD5_SUM = 'eb9058c3a382ffc7106e4002c42a8d85'

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str,
    default=os.environ.get('TF_DATA_DIR', '/tmp/cifar100_data'),
    help='Directory to download data and extract the tarball')


def unpickle(file):
    if sys.version_info >= (3, 0):
        data_dict = cPickle.load(file, encoding='bytes')
    else:
        data_dict = cPickle.load(file)
    return data_dict


def create_writer(subset):
    return tf.python_io.TFRecordWriter(
        os.path.join(FLAGS.data_dir, 'cifar100_{}.tfrecords'.format(subset)))


def examples_in_tar_file(tar_file, data_file):
    tar_file_obj = tarfile.open(tar_file, 'r:gz')
    for file_info in tar_file_obj:
        if data_file in file_info.name:
            data_dict = unpickle(tar_file_obj.extractfile(file_info))
            if sys.version_info >= (3, 0):
                data = data_dict[b'data']
                fine_labels = data_dict[b'fine_labels']
                coarse_labels = data_dict[b'coarse_labels']
            else:
                data = data_dict['data']
                fine_labels = data_dict['fine_labels']
                coarse_labels = data_dict['coarse_labels']

            for image, fine_label, coarse_label in zip(data, fine_labels, coarse_labels):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                        'fine_label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[fine_label])),
                        'coarse_label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[coarse_label]))
                    }))

                yield example


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

    with create_writer('train') as train_writer:
        with create_writer('validation') as val_writer:
            example_count = 0
            train_example_count = 0
            validation_example_count = 0
            for example in examples_in_tar_file(file_path, 'train'):
                if example_count < 40000:
                    train_writer.write(example.SerializeToString())
                    train_example_count += 1
                else:
                    val_writer.write(example.SerializeToString())
                    validation_example_count += 1

                example_count += 1

    print('Write %d examples from %s to %s.tfrecords' % (
        train_example_count, 'train', 'train'
    ))
    print('Write %d examples from %s to %s.tfrecords' % (
        validation_example_count, 'train', 'validation'
    ))

    with create_writer('test') as writer:
        example_count = 0
        for example in examples_in_tar_file(file_path, 'test'):
            writer.write(example.SerializeToString())
            example_count += 1
        print('Write %d examples from %s to %s.tfrecords' % (
            example_count, 'test', 'test'
        ))


if __name__ == '__main__':
    FLAGS, not_parsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + not_parsed)
