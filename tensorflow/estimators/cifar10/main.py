#!/bin/env python3
"""CIFAR-10 estimator runner.
"""
import os
import tensorflow as tf

from cnn_estimator import CNNEstimator
from vgg_estimator import VggEstimator

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
EVAL_FILE = 'eval.tfrecords'
FLAGS = tf.app.flags.FLAGS


tf.logging.set_verbosity(tf.logging.INFO)

def parser(record):
    keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed['label'], tf.int32)
    return image, label


def input_fn(filenames, is_training=True):
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(parser)  # Parse the record into tensors.
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000) # Shuffle the dataset
        dataset = dataset.repeat()  # Repeat the input indefinitely.
        dataset = dataset.batch(32)
    else:
        dataset = dataset.repeat(1)
        dataset = dataset.batch(10000)

    features, labels = dataset.make_one_shot_iterator().get_next()

    return features, labels


def main(unused_argv):
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    estimator = VggEstimator({
        'feature_columns': [tf.feature_column.numeric_column('image')],
    })

    if FLAGS.mode.lower() == 'train':
        estimator.train(input_fn=lambda: input_fn(os.path.join(FLAGS.data_dir, TRAIN_FILE)))
    elif FLAGS.mode.lower() == 'validation':
        eval_results = estimator.evaluate(input_fn=lambda: input_fn(os.path.join(FLAGS.data_dir, VALIDATION_FILE), False))
        print(eval_results)
    elif FLAGS.mode.lower() == 'predict':
        estimator.predict(input_fn=lambda: input_fn(os.path.join(FLAGS.data_dir, EVAL_FILE), False))
    else:
        print("Unknown mode: %s" % FLAGS.mode)

if __name__ == '__main__':
    tf.app.flags.DEFINE_string("model_dir",
                                "/tmp/cifar10_model",
                                "Specify the model dir.")
    tf.app.flags.DEFINE_string("data_dir",
                                "/tmp/cifar10_data",
                                "Specify the data dir.")
    tf.app.flags.DEFINE_string("mode",
                                "train",
                                "Runing mode.")
    tf.app.flags.DEFINE_float("learning_rate",
                                0.001,
                                "Learning rate.")
    tf.app.run()
