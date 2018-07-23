# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Contains utility and supporting functions for Model.

This module contains code which does not directly build model layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the model layers can be found in models/*_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf


def build_model_fn(features, labels, mode, model,
                   weight_decay, momentum, loss_scale,
                   learning_rate_fn, loss_filter_fn=None,
                   dtype=tf.float32):
    """Shared functionality for different model_fns.

      Initializes the Model representing the model layers
      and uses that model to build the necessary EstimatorSpecs for
      the `mode` in question. For training, this means building losses,
      the optimizer, and the train op that get passed into the EstimatorSpec.
      For evaluation and prediction, the EstimatorSpec is returned without
      a train op, but with the necessary parameters for the given mode.

      Args:
        features: tensor representing input images
        labels: tensor representing class labels for all input images
        mode: current estimator mode; should be one of
          `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
        model: the model representing a TensorFlow model that has a __call__
          function. We assume here that this is a subclass of ResnetModel.
        weight_decay: weight decay loss rate used to regularize learned variables.
        learning_rate_fn: function that returns the current learning rate given
          the current global_step
        momentum: momentum term used for optimization
          If set to None, the format is dependent on whether a GPU is available.
          use. See README for details. Valid values: [1, 2]
        loss_scale: The factor to scale the loss for numerical stability. A detailed
          summary is present in the arg parser help text.
        loss_filter_fn: function that takes a string variable name and returns
          True if the var should be included in loss calculation, and False
          otherwise. If None, batch_normalization variables will be excluded
          from the loss.
        dtype: the TensorFlow dtype to use for calculations.

      Returns:
        EstimatorSpec parameterized according to the input params and the
        current mode.
    """

    # Generate a summary node for the images
    tf.summary.image('images', features, max_outputs=6)

    features = tf.cast(features, dtype)

    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is is low precision, logits must be cast to
    # fp32 for numerical stability.
    logits = tf.cast(logits, tf.float32)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    cross_entropy = tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32))
         for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        learning_rate = tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum
        )

        if loss_scale != 1:
            # When computing fp16 gradients, often intermediate tensor values are
            # so small, they underflow to 0. To avoid this, we multiply the loss by
            # loss_scale to make these tensor values loss_scale times bigger.
            scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

            # Once the gradient computation is complete we can scale the gradients
            # back to the correct scale before passing them to the optimizer.
            unscaled_grad_vars = [(grad / loss_scale, var)
                                  for grad, var in scaled_grad_vars]
            minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
        else:
            minimize_op = optimizer.minimize(loss, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    if not tf.contrib.distribute.has_distribution_strategy():
        accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    else:
        # Metrics are currently not compatible with distribution strategies during
        # training. This does not affect the overall performance of the model.
        # accuracy = (tf.no_op(), tf.constant(0))
        accuracy = tf.metrics.accuracy(labels, predictions['classes'])

    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def main_loop(flags_obj, model_name, model_function,
              input_function, dataset_name, shape=None, estimator=None):
    """Shared main loop for Models.

      Args:
        flags_obj: An object containing parsed flags. See define_resnet_flags()
          for details.
        model_name: the name of the running model.
        model_function: the function that instantiates the Model and builds the
          ops for train/eval. This will be passed directly into the estimator.
        input_function: the function that processes the dataset and returns a
          dataset that the estimator can train on. This will be wrapped with
          all the relevant flags for running and passed to estimator.
        dataset_name: the name of the dataset for training and evaluation. This is
          used for logging purpose.
        shape: list of ints representing the shape of the images used for training.
          This is only used if flags_obj.export_dir is passed.
        estimator: The tf.estimator.Estimator instance. One can use the Keras converted model here.
    """

    # model_helpers.apply_clean(flags.FLAGS)

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Create session config based on values of inter_op_parallelism_threads and
    # intra_op_parallelism_threads. Note that we default to having
    # allow_soft_placement = True, which is required for multi-GPU and not
    # harmful for other modes.
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
        intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
        allow_soft_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 1.0

    num_gpus = flags_core.get_num_gpus(flags_obj)
    distribution_strategy = distribution_utils.get_distribution_strategy(
        num_gpus, flags_obj.all_reduce_alg)

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy, session_config=session_config)

    if estimator is None:
        classifier = tf.estimator.Estimator(
            model_fn=model_function,
            model_dir=flags_obj.model_dir,
            config=run_config,
            params={
                'data_format': flags_obj.data_format,
                'batch_size': flags_obj.batch_size,
                'loss_scale': flags_core.get_loss_scale(flags_obj),
                'dtype': flags_core.get_tf_dtype(flags_obj)
            })
    else:
        classifier = estimator

    run_params = {
        'batch_size': flags_obj.batch_size,
        'dtype': flags_core.get_tf_dtype(flags_obj),
        'synthetic_data': flags_obj.use_synthetic_data,
        'train_epochs': flags_obj.train_epochs,
    }
    if flags_obj.use_synthetic_data:
        dataset_name = dataset_name + '-synthetic'

    benchmark_logger = logger.get_benchmark_logger()
    benchmark_logger.log_run_info(model_name, dataset_name, run_params,
                                  test_id=flags_obj.benchmark_test_id)

    train_hooks = hooks_helper.get_train_hooks(
        flags_obj.hooks,
        model_dir=flags_obj.model_dir,
        batch_size=flags_obj.batch_size)

    def input_fn_train():
        return input_function(
            is_training=True, data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, num_gpus),
            num_epochs=flags_obj.epochs_between_evals,
            num_gpus=num_gpus)

    def input_fn_eval():
        return input_function(
            is_training=False, data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, num_gpus),
            num_epochs=1)

    total_training_cycle = (flags_obj.train_epochs //
                            flags_obj.epochs_between_evals)
    for cycle_index in range(total_training_cycle):
        tf.logging.info('Starting a training cycle: %d/%d',
                        cycle_index, total_training_cycle)

        classifier.train(input_fn=input_fn_train, hooks=train_hooks,
                         max_steps=flags_obj.max_train_steps)

        tf.logging.info('Starting to evaluate.')

        # flags_obj.max_train_steps is generally associated with testing and
        # profiling. As a result it is frequently called with synthetic data, which
        # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
        # eval (which is generally unimportant in those circumstances) to terminate.
        # Note that eval will run for max_train_steps each loop, regardless of the
        # global_step count.
        eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                           steps=flags_obj.max_train_steps)

        benchmark_logger.log_evaluation_result(eval_results)

        if model_helpers.past_stop_threshold(
                flags_obj.stop_threshold,
                eval_results['accuracy']):
            break

    if flags_obj.export_dir is not None:
        # Exports a saved model for the given classifier.
        input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
            shape, batch_size=flags_obj.batch_size)
        classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn)
