"""Util functions
"""
import tensorflow as tf


def build_model_fn(construct_model, resize_image=True):
    def _model_fn(features, labels, mode, params):
        """Estimator model function.

        Arguments:
            features {Tensor} -- features in input
            labels {Tensor} -- labels of features
            mode {tf.estimator.ModeKeys} -- mode key
            params {any} -- model params
        """
        # Use `input_layer` to apply the feature columns.
        if resize_image:
            input_layer = tf.image.resize_images(features['image'], [224, 224])
        else:
            input_layer = tf.reshape(features['image'], [-1, 32, 32, 3])

        logits = construct_model(input_layer, mode == tf.estimator.ModeKeys.TRAIN, params=params)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        export_outputs = {
            'predict_output': tf.estimator.export.PredictOutput(predictions)
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        accuracy_mean = tf.reduce_mean(accuracy, name='accuracy_mean')
        tf.summary.scalar('accuracy', accuracy_mean)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
            # optimizer = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=0.9)
            # optimizer = tf.train.AdagradOptimizer(params['learning_rate'])
            optimizer = tf.train.AdamOptimizer(params['learning_rate'])

            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = { "accuracy": accuracy }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return _model_fn
