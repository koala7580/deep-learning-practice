"""Base Model
"""
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge


class BaseModel(object):

    def __init__(self):
        self._swap_out_ts = []

    def remap_gradients(self):
        for t in self._swap_out_ts:
            t_name = t.name.rsplit('/', 1)[0]

            with tf.device('/cpu:0'):
                on_cpu = tf.identity(t, '%s/on_cpu' % t_name)

            for op in t.consumers():
                for i, tensor in enumerate(op.inputs):
                    if tensor == t and op.name.startswith('gradients'):
                        op._update_input(i, on_cpu, False)

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
