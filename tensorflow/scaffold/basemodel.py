# -*- coding: utf-8 -*-
"""Base class for building a network, which sharing inference, traning and loss functions.

 Python 3.x needed.
"""
# pylint: disable=bad-indentation
import re
import tensorflow as tf


class BaseModel:
  """Base class for a tensorflow model.

  Summary of available functions:

  # Compute input images and labels for training. If you would like to run
  # evaluations, use inputs() instead.
  inputs, labels = distorted_inputs()

  # Compute inference on the model inputs to make a prediction.
  predictions = inference(inputs)

  # Compute the total loss of the prediction with respect to the labels.
  loss = loss(predictions, labels)

  # Create a graph to run one step of training with respect to the loss.
  train_op = train(loss, global_step)
  """
  FLAGS = tf.app.flags.FLAGS

  # If a model is trained with multiple GPU's prefix all Op names with tower_name
  # to differentiate the operations. Note that this prefix is removed from the
  # names of the summaries when visualizing a model.
  TOWER_NAME = 'tower'

  # Global constants describing the CIFAR-10 data set.
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
  NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0

  # Constants describing the training process.
  MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
  NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
  LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
  INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


  def __init__(self):
    """Init the class."""
    # Basic model parameters.
    tf.app.flags.DEFINE_integer('batch_size', 128,
                                """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_string('data_dir', '/tmp/tf_data',
                               """Path to the model data directory.""")


  def _activation_summary(self, x): # pylint: disable=invalid-name
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % self.TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


  def _variable_on_cpu(self, name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
    return var


  def _variable_with_weight_decay(self, name, shape, stddev, wd): # pylint: disable=invalid-name
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = self._variable_on_cpu(name, shape,
                                tf.truncated_normal_initializer(stddev=stddev))
    if wd:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    return var


  def _add_loss_summaries(self, total_loss):
    """Add summaries for losses in the model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]: # pylint: disable=invalid-name
      # Name each loss as '(raw)' and name the moving average version of the loss
      # as the original loss name.
      tf.summary.scalar(l.op.name +' (raw)', l)
      tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


  def distorted_inputs(self):
    """Construct distorted input for model training."""
    raise NotImplementedError('Please supply a `distorted_inputs` implementation')


  def inputs(self, eval_data):
    """Construct input for model evaluation.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
    """
    raise NotImplementedError('Please supply a `inputs` implementation')


  def inference(self, images):
    """Build the model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    raise NotImplementedError('Please supply a `inference` implementation')


  def loss(self, logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


  def train(self, total_loss, global_step):
    """Train the model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / self.FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE, # pylint: disable=invalid-name
                                    global_step,
                                    decay_steps,
                                    self.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = self._add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.GradientDescentOptimizer(lr)
      grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        self.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op
