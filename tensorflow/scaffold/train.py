# -*- coding: utf-8 -*-
"""Train using a single GPU.
"""
# pylint: disable=bad-indentation
import time
import os.path
from datetime import datetime

import numpy as np
import tensorflow as tf


# IMPORT model
from cifar10 import CIFAR10Model


def train(model):
  """Train model for a number of steps."""
  with tf.Graph().as_default(): # pylint: disable=not-context-manager
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for the model.
    images, labels = model.train_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images)

    # Calculate loss.
    loss = model.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=model.FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(model.FLAGS.train_dir,
                                           graph=sess.graph)

    for step in range(model.FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = model.FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step, loss_value,
                            examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == model.FLAGS.max_steps:
        checkpoint_path = os.path.join(model.FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  """Main Function"""
  cifar10 = CIFAR10Model()

  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(cifar10.FLAGS.train_dir):
    tf.gfile.DeleteRecursively(cifar10.FLAGS.train_dir)
  tf.gfile.MakeDirs(cifar10.FLAGS.train_dir)

  train(cifar10)


if __name__ == '__main__':
  tf.app.flags.DEFINE_string('train_dir', '/tmp/tf_train',
                             """Directory where to write event logs """
                             """and checkpoint.""")
  tf.app.flags.DEFINE_integer('max_steps', 1000000,
                              """Number of batches to run.""")
  tf.app.flags.DEFINE_boolean('log_device_placement', False,
                              """Whether to log device placement.""")

  tf.app.run()
