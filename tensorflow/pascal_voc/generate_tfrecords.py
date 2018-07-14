"""Convert the Pascal VOC dataset of Classification/Detect to .tfrecords file.
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import tensorflow as tf
import skimage.io as skimage_io
from lxml import etree

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
CLASSES.sort()


def write_set(data_set, image_dir, annotation_dir, writer):
    for index, item in enumerate(data_set):
        with open(os.path.join(annotation_dir, item + '.xml')) as f:
            xml_content = f.read()

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

        image = skimage_io.imread(os.path.join(image_dir, image_filename[0]))

        image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
        image_shape_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[
            int(image_height[0]),
            int(image_width[0]),
            int(image_depth[0]),
        ]))
        objects_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.text.encode('utf-8') for x in objects_name]))

        bbox_xmin_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x) for x in bbox_xmin]))
        bbox_xmax_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x) for x in bbox_xmax]))
        bbox_ymin_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x) for x in bbox_ymin]))
        bbox_ymax_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x) for x in bbox_ymax]))

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': image_feature,
                'image_shape': image_shape_feature,
                'objects': objects_feature,
                'bbox_xmin': bbox_xmin_feature,
                'bbox_xmax': bbox_xmax_feature,
                'bbox_ymin': bbox_ymin_feature,
                'bbox_ymax': bbox_ymax_feature,
            }
        ))

        writer.write(example.SerializeToString())
        print('convert {}/{} items'.format(index, len(data_set)), end='\r')

    print('\nConvert a dataset.')


def main(opt):
    if (not os.path.exists(opt.src_dir) or
        not os.path.exists(os.path.join(opt.src_dir, 'ImageSets')) or
        not os.path.exists(os.path.join(opt.src_dir, 'JPEGImages')) or
        not os.path.exists(os.path.join(opt.src_dir, 'Annotations'))):
        raise RuntimeError('%s is not a Pascal VOC dataset dir.' % opt.src_dir)

    image_dir = os.path.join(opt.src_dir, 'JPEGImages')
    annotation_dir = os.path.join(opt.src_dir, 'Annotations')

    # load train set
    with open(os.path.join(opt.src_dir, 'ImageSets', 'Main', 'train.txt')) as fp:
        train_set = [line.strip() for line in fp.readlines()]

    with tf.python_io.TFRecordWriter(os.path.join(opt.dst_dir, 'pascal_voc_train.tfrecords')) as writer:
        write_set(train_set, image_dir, annotation_dir, writer)

    # load eval set
    with open(os.path.join(opt.src_dir, 'ImageSets', 'Main', 'val.txt')) as fp:
        eval_set = [line.strip() for line in fp.readlines()]

    with tf.python_io.TFRecordWriter(os.path.join(opt.dst_dir, 'pascal_voc_eval.tfrecords')) as writer:
        write_set(eval_set, image_dir, annotation_dir, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tfrecords from Pascal VOC dataset.')
    parser.add_argument(
        '-s', '--src-dir',
        type=str,
        required=True,
        help='The source directory of VOC2012'
    )
    parser.add_argument(
        '-d', '--dst-dir',
        type=str,
        required=True,
        help='The distinate directory of tfrecords file.'
    )

    main(parser.parse_args())
