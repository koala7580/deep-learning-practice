# -*- coding: utf-8 -*-
"""Evaluation for the model.
"""
import math
import time
from datetime import datetime

import numpy as np
import tensorflow as tf


# Import the model
from cifar10 import CIFAR10Model


def eval_once(model, saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    model: Model instance.
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model.FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS): # pylint: disable=invalid-name
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(model.FLAGS.num_examples / model.FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * model.FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except,invalid-name
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(model):
  """Eval model for a number of steps."""
  with tf.Graph().as_default(): # pylint: disable=not-context-manager
    # Get images and labels for CIFAR-10.
    eval_data = model.FLAGS.eval_data == 'test'
    images, labels = model.evaluate_inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    graph = tf.get_default_graph()
    summary_writer = tf.summary.FileWriter(model.FLAGS.eval_dir,
                                           graph=graph)

    while True:
      eval_once(model, saver, summary_writer, top_k_op, summary_op)
      if model.FLAGS.run_once:
        break
      time.sleep(model.FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  """Main Function"""
  cifar10 = CIFAR10Model()
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(cifar10.FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(cifar10.FLAGS.eval_dir)
  tf.gfile.MakeDirs(cifar10.FLAGS.eval_dir)
  evaluate(cifar10)


if __name__ == '__main__':
  tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                             """Directory where to write event logs.""")
  tf.app.flags.DEFINE_string('eval_data', 'test',
                             """Either 'test' or 'train_eval'.""")
  tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                             """Directory where to read model checkpoints.""")
  tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                              """How often to run the eval.""")
  tf.app.flags.DEFINE_integer('num_examples', 10000,
                              """Number of examples to run.""")
  tf.app.flags.DEFINE_boolean('run_once', False,
                              """Whether to run eval only once.""")

  tf.app.run()
