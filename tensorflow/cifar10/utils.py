"""Util functions
"""
import tensorflow as tf


def build_model_fn(args, build_model):
    def _model_fn(features, labels, mode, params):
        """Estimator model function.

        Arguments:
            features {Tensor} -- features in input
            labels {Tensor} -- labels of features
            mode {tf.estimator.ModeKeys} -- mode key
            params {any} -- model params
        """
        inputs = features['image']
        tf.summary.image('input_image', inputs)

        logits = build_model(inputs, args, mode, params)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            lr = tf.train.exponential_decay(
                    args.learning_rate,
                    tf.train.get_global_step(),
                    30000,
                    0.1)
            # optimizer = tf.train.GradientDescentOptimizer(lr)
            optimizer = tf.train.AdamOptimizer(args.learning_rate)

            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops={ "accuracy": accuracy })

    return _model_fn
