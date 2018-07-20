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
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf

from app.utils.export import export
from app.utils.logs import hooks_helper
from app.utils.logs import logger
from app.utils.misc import distribution_utils
from app.utils.misc import model_helpers


# pylint: enable=g-bad-import-order

################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
        batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
    """Get a learning rate that decays step-wise as training progresses.

      Args:
        batch_size: the number of examples processed in each training batch.
        batch_denom: this value will be used to scale the base learning rate.
          `0.1 * batch size` is divided by this number, such that when
          batch_denom == batch_size, the initial learning rate will be 0.1.
        num_images: total number of images that will be used for training.
        boundary_epochs: list of ints representing the epochs at which we
          decay the learning rate.
        decay_rates: list of floats representing the decay rates to be used
          for scaling the learning rate. It should have one more element
          than `boundary_epochs`, and all elements should have the same type.

      Returns:
        Returns a function that takes a single argument - the number of batches
        trained so far (global_step)- and returns the learning rate to be used
        for training the next batch.
    """
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Reduce the learning rate at certain epochs.
    # CIFAR-10: divide by 10 at epoch 100, 150, and 200
    # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    values = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, values)

    return learning_rate_fn


def model_fn(features, labels, mode, model,
             weight_decay, learning_rate_fn,
             momentum, loss_scale,
             loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE):
    """Shared functionality for different resnet model_fns.

      Initializes the ResnetModel representing the model layers
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
        model: a class representing a TensorFlow model that has a __call__
          function. We assume here that this is a subclass of ResnetModel.
        weight_decay: weight decay loss rate used to regularize learned variables.
        learning_rate_fn: function that returns the current learning rate given
          the current global_step
        momentum: momentum term used for optimization
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
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
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


def main(config_obj, model_function, input_function, dataset_name, shape=None):
    """Shared main loop for ResNet Models.

      Args:
        config_obj: An object containing run config.
        model_function: the function that instantiates the Model and builds the
          ops for train/eval. This will be passed directly into the estimator.
        input_function: the function that processes the dataset and returns a
          dataset that the estimator can train on. This will be wrapped with
          all the relevant flags for running and passed to estimator.
        dataset_name: the name of the dataset for training and evaluation. This is
          used for logging purpose.
        shape: list of ints representing the shape of the images used for training.
          This is only used if flags_obj.export_dir is passed.
    """

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Create session config based on values of inter_op_parallelism_threads and
    # intra_op_parallelism_threads. Note that we default to having
    # allow_soft_placement = True, which is required for multi-GPU and not
    # harmful for other modes.
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=config_obj.inter_op_parallelism_threads,
        intra_op_parallelism_threads=config_obj.intra_op_parallelism_threads,
        allow_soft_placement=True)

    # distribution_strategy = distribution_utils.get_distribution_strategy(
    #     flags_core.get_num_gpus(config_obj), config_obj.all_reduce_alg)
    distribution_strategy = None

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy, session_config=session_config)

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=config_obj.model_dir, config=run_config,
        params={
            'resnet_size': int(config_obj.resnet_size),
            'data_format': config_obj.data_format,
            'batch_size': config_obj.batch_size,
            'resnet_version': int(config_obj.resnet_version),
            'loss_scale': flags_core.get_loss_scale(config_obj),
            'dtype': flags_core.get_tf_dtype(config_obj)
        })

    run_params = {
        'batch_size': config_obj.batch_size,
        'dtype': flags_core.get_tf_dtype(config_obj),
        'resnet_size': config_obj.resnet_size,
        'resnet_version': config_obj.resnet_version,
        'synthetic_data': config_obj.use_synthetic_data,
        'train_epochs': config_obj.train_epochs,
    }
    if config_obj.use_synthetic_data:
        dataset_name = dataset_name + '-synthetic'

    benchmark_logger = logger.get_benchmark_logger()
    benchmark_logger.log_run_info('resnet', dataset_name, run_params,
                                  test_id=config_obj.benchmark_test_id)

    train_hooks = hooks_helper.get_train_hooks(
        config_obj.hooks,
        model_dir=config_obj.model_dir,
        batch_size=config_obj.batch_size)

    def input_fn_train():
        return input_function(
            is_training=True, data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            num_epochs=flags_obj.epochs_between_evals,
            num_gpus=flags_core.get_num_gpus(flags_obj))

    def input_fn_eval():
        return input_function(
            is_training=False, data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            num_epochs=1)

    total_training_cycle = (config_obj.train_epochs //
                            config_obj.epochs_between_evals)
    for cycle_index in range(total_training_cycle):
        tf.logging.info('Starting a training cycle: %d/%d',
                        cycle_index, total_training_cycle)

        classifier.train(input_fn=input_fn_train, hooks=train_hooks,
                         max_steps=config_obj.max_train_steps)

        tf.logging.info('Starting to evaluate.')

        # flags_obj.max_train_steps is generally associated with testing and
        # profiling. As a result it is frequently called with synthetic data, which
        # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
        # eval (which is generally unimportant in those circumstances) to terminate.
        # Note that eval will run for max_train_steps each loop, regardless of the
        # global_step count.
        eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                           steps=config_obj.max_train_steps)

        benchmark_logger.log_evaluation_result(eval_results)

        if model_helpers.past_stop_threshold(
                config_obj.stop_threshold, eval_results['accuracy']):
            break

    if config_obj.export_dir is not None:
        # Exports a saved model for the given classifier.
        input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
            shape, batch_size=config_obj.batch_size)
        classifier.export_savedmodel(config_obj.export_dir, input_receiver_fn)


def define_resnet_flags(resnet_size_choices=None):
    """Add flags and validators for ResNet."""
    flags_core.define_base()
    flags_core.define_performance(num_parallel_calls=False)
    flags_core.define_image()
    flags_core.define_benchmark()
    flags.adopt_module_key_flags(flags_core)

    flags.DEFINE_enum(
        name='resnet_version', short_name='rv', default='2',
        enum_values=['1', '2'],
        help=flags_core.help_wrap(
            'Version of ResNet. (1 or 2) See README.md for details.'))

    choice_kwargs = dict(
        name='resnet_size', short_name='rs', default='50',
        help=flags_core.help_wrap('The size of the ResNet model to use.'))

    if resnet_size_choices is None:
        flags.DEFINE_string(**choice_kwargs)
    else:
        flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)

    # The current implementation of ResNet v1 is numerically unstable when run
    # with fp16 and will produce NaN errors soon after training begins.
    msg = ('ResNet version 1 is not currently supported with fp16. '
           'Please use version 2 instead.')

    @flags.multi_flags_validator(['dtype', 'resnet_version'], message=msg)
    def _forbid_v1_fp16(flag_values):  # pylint: disable=unused-variable
        return (flags_core.DTYPE_MAP[flag_values['dtype']][0] != tf.float16 or
                flag_values['resnet_version'] != '1')
