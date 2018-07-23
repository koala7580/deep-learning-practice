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

"""Downloads and convert the Pascal VOC dataset to TFRecord."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import skimage.io

from lxml import etree
from practice.utils.check_sum import file_md5_check

DATA_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
MD5_SUM = '6cd6e144f989b92b3379bac3b3de84fd'

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str,
    default=os.environ.get('TF_DATA_DIR', '/tmp/pascal_voc_data'),
    help='Directory to download data and extract the tarball')


def file_info(tar_file_obj, predicate):
    for info in tar_file_obj:
        if predicate(info.name):
            return info


def examples_in_tar_file(tar_file_obj, records):
    for index, record in enumerate(records):
        annotation_info = file_info(tar_file_obj,
                                    lambda name: ('%s.xml' % record) in name)
        with tar_file_obj.extractfile(annotation_info) as fo:
            xml_content = fo.read()

        annotation_xml = etree.fromstring(xml_content)
        image_filename = annotation_xml.xpath('/annotation/filename/text()')

        image_width = annotation_xml.xpath('/annotation/size/width/text()')
        image_height = annotation_xml.xpath('/annotation/size/height/text()')
        image_depth = annotation_xml.xpath('/annotation/size/depth/text()')

        objects_name = annotation_xml.xpath('/annotation/object/name')

        bbox_xmin = annotation_xml.xpath('/annotation/object/bndbox/xmin/text()')
        bbox_ymin = annotation_xml.xpath('/annotation/object/bndbox/ymin/text()')
        bbox_xmax = annotation_xml.xpath('/annotation/object/bndbox/xmax/text()')
        bbox_ymax = annotation_xml.xpath('/annotation/object/bndbox/ymax/text()')

        image_info = file_info(tar_file_obj,
                               lambda name: ('JPEGImages/%s' % image_filename[0]) in name)

        with tar_file_obj.extractfile(image_info) as fo:
            image = skimage.io.imread(fo)

        image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            image.tobytes()
        ]))
        image_shape_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[
            int(image_height[0]),
            int(image_width[0]),
            int(image_depth[0]),
        ]))
        objects_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            x.text.encode('utf-8') for x in objects_name
        ]))

        bbox_xmin_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[
            int(x) for x in bbox_xmin
        ]))
        bbox_xmax_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[
            int(x) for x in bbox_xmax
        ]))
        bbox_ymin_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[
            int(x) for x in bbox_ymin
        ]))
        bbox_ymax_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[
            int(x) for x in bbox_ymax
        ]))

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': image_feature,
                'image/shape': image_shape_feature,
                'objects': objects_feature,
                'bbox/xmin': bbox_xmin_feature,
                'bbox/xmax': bbox_xmax_feature,
                'bbox/ymin': bbox_ymin_feature,
                'bbox/ymax': bbox_ymax_feature,
            }
        ))

        yield example
        print('>> Convert %d/%d' % (index, len(records)), end='\r')


def create_writer(subset):
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    return tf.python_io.TFRecordWriter(
        os.path.join(FLAGS.data_dir, 'pascal_voc_{}.tfrecords'.format(subset)),
        options=options
    )


def main(_):
    """Download and convert the Pascal VOC dataset into TFRecord."""
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

    with tarfile.open(file_path, 'r') as tar_file_obj:
        for info in tar_file_obj:
            if 'ImageSets/Main/train.txt' in info.name:
                with tar_file_obj.extractfile(info) as fo:
                    train_files = [line.decode('utf-8').strip() for line in fo.readlines()]

                with create_writer('train') as writer:
                    example_count = 0
                    for example in examples_in_tar_file(tar_file_obj, train_files):
                        writer.write(example.SerializeToString())
                        example_count += 1

                    print('\nWrite %d examples to %s.tfrecords' % (
                        example_count, 'train'
                    ))

            if 'ImageSets/Main/val.txt' in info.name:
                with tar_file_obj.extractfile(info) as fo:
                    val_files = [line.decode('utf-8').strip() for line in fo.readlines()]

                with create_writer('validation') as writer:
                    example_count = 0
                    for example in examples_in_tar_file(tar_file_obj, val_files):
                        writer.write(example.SerializeToString())
                        example_count += 1

                    print('\nWrite %d examples to %s.tfrecords' % (
                        example_count, 'validation'
                    ))


if __name__ == '__main__':
    FLAGS, not_parsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + not_parsed)
