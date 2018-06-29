"""Base Model
"""
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge


class BaseModel(object):

    def __init__(self, batch_size):
        self._batch_size = batch_size
        self._swap_out_variables = []

    def _swap_out(self, x):
        with tf.device('/cpu:0'):
            x_name = x.name.rsplit('/', 1)[0]

            var_shape = x.get_shape().as_list()
            var_shape[0] = self._batch_size
            swap_out_var = tf.get_variable('%s/swap_var' % x_name,
                                            shape=var_shape,
                                            dtype=x.dtype,
                                            trainable=False,
                                            initializer=tf.zeros_initializer)
            swap_out_op = tf.assign(swap_out_var, x, name='%s/swap_out' % x_name)

            self._swap_out_variables.append((x, swap_out_op))

            return swap_out_op

    def modity_gradients(self, graph):
        ts0 = [v[0] for v in self._swap_out_variables]
        ts1 = [v[1] for v in self._swap_out_variables]
        svg = ge.sgv('gradients', graph=graph)
        n = ge.swap_ts(ts0, ts1, can_modify=svg.ops)

        tf.logging.info('%d tensors swapped in gradients subgraph view.' % n)

    def _detect_data_format(self, image):
        shape = image.get_shape()
        if shape[1] == 3:
            return 'channels_first'
        else:
            return 'channels_last'

    def _transform_data_format(self, x, from_data_format, to_data_format):
        assert from_data_format in ['channels_first', 'channels_last']
        assert to_data_format in ['channels_first', 'channels_last']

        if from_data_format != to_data_format:
            if from_data_format == 'channels_last':
                # Computation requires channels_first.
                x = tf.transpose(x, [0, 3, 1, 2])
            else:
                # Computation requires channels_last.
                x = tf.transpose(x, [0, 2, 3, 1])

        return x
