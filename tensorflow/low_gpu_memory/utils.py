"""Util functions
"""
import os
import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs
from tensorflow.python.training import training_util
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer

def build_model_fn(build_model, build_input_layer, args):
    def _model_fn(features, labels, mode, params):
        """Estimator model function.

        Arguments:
            features {Tensor} -- features in input
            labels {Tensor} -- labels of features
            mode {tf.estimator.ModeKeys} -- mode key
            params {any} -- model params
        """
        input_layer = build_input_layer(features)

        tf.summary.image('image', input_layer)

        logits, instance = build_model(input_layer, mode == tf.estimator.ModeKeys.TRAIN, params=params, args=args)

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
        
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
            optimizer = tf.train.AdamOptimizer(args.learning_rate)

            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            
            instance.remap_gradients()

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops={ "accuracy": accuracy })

    return _model_fn


class MetadataHook(SessionRunHook):
    def __init__ (self,
                  save_steps=None,
                  save_secs=None,
                  output_dir=""):
        self._output_tag = "step-{}"
        self._output_dir = output_dir
        self._timer = SecondOrStepTimer(
            every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        self._next_step = None
        self._global_step_tensor = training_util.get_global_step()
        self._writer = tf.summary.FileWriter (os.path.join(self._output_dir, 'train'), tf.get_default_graph())

        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use ProfilerHook.")

    def before_run(self, run_context):
        self._request_summary = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step)
        )
        requests = {"global_step": self._global_step_tensor}
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            if self._request_summary else None)
        return SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._writer.add_run_metadata(
                run_values.run_metadata, self._output_tag.format(global_step))
            self._writer.flush()
        self._next_step = global_step + 1

    def end(self, session):
        self._writer.close()
