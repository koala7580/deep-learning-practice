# coding: utf-8
# Low Memory VGG model.

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import timeline

tf.logging.set_verbosity(tf.logging.DEBUG)


BATCH_SIZE = 32

class VGG(object):
    def __init__(self):
        self._transfer_ts = []
        self._data_format = 'channels_first'

    def build_model(self):
        inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='images')
        labels = tf.placeholder(tf.int32, [None, 1], name='labels')

        if self._data_format == 'channels_first':
            with tf.device('/cpu:0'):
                x = tf.transpose(inputs, [0, 3, 1, 2])
        else:
            x = inputs

        # Block1
        x = self._conv_layer(x, 64, 'block1_conv1')
        x = self._conv_layer(x, 64, 'block1_conv2')
        x = self._max_pool_layer(x, 'block1_pool')

        # Block2
        x = self._conv_layer(x, 128, 'block2_conv1')
        x = self._conv_layer(x, 128, 'block2_conv2')
        x = self._max_pool_layer(x, 'block2_pool')

        # Block3
        x = self._conv_layer(x, 256, 'block3_conv1')
        x = self._conv_layer(x, 256, 'block3_conv2')
        x = self._conv_layer(x, 256, 'block3_conv3')
        x = self._max_pool_layer(x, 'block3_pool')

        # Block4
        x = self._conv_layer(x, 512, 'block4_conv1')
        x = self._conv_layer(x, 512, 'block4_conv2')
        x = self._conv_layer(x, 512, 'block4_conv3')
        x = self._max_pool_layer(x, 'block4_pool')

        # Block5
        x = self._conv_layer(x, 512, 'block5_conv1')
        x = self._conv_layer(x, 512, 'block5_conv2')
        x = self._conv_layer(x, 512, 'block5_conv3')
        x = self._max_pool_layer(x, 'block5_pool')

        # Classification block
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=1024, activation=tf.nn.relu, name='fc1')
        x = tf.layers.dense(x, units=1024, activation=tf.nn.relu, name='fc2')
        x = tf.layers.dense(x, units=2, name='logits')

        return inputs, labels, x

    def remap_gradients(self, graph):
        for t in self._transfer_ts:
            t_name = t.name.rsplit('/', 1)[0]
            with tf.device('/cpu:0'):
                identity = tf.identity(t, name='%s/transfer_to_cpu' % t_name)

            for op in t.consumers():
                for i, tensor in enumerate(op.inputs):
                    # if tensor == t and op != identity.op:
                    if tensor == t and op.name.startswith('gradients'):
                        op._update_input(i, identity, False)

    def _conv_layer(self, x, filters, name):
        layer = tf.layers.conv2d(x, filters, 3, padding='same', activation=tf.nn.relu, data_format=self._data_format, name=name)
        self._transfer_ts.append(layer)
        return layer

    def _max_pool_layer(self, x, name):
        layer = tf.layers.max_pooling2d(x, pool_size=2, strides=2, data_format=self._data_format, name=name)
        self._transfer_ts.append(layer)
        return layer


def main():
    vgg = VGG()
    inputs, labels, logits = vgg.build_model()
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss, name='train_op')

    vgg.remap_gradients(tf.get_default_graph())

    os.system('rm -rf /tmp/vgg')

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        train_writer = tf.summary.FileWriter('/tmp/vgg/train', sess.graph)

        summary_op = tf.summary.merge_all()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())

        # Train loop
        for i in range(5):
            image_batch = np.random.rand(BATCH_SIZE, 224, 224, 3)
            label_batch = np.ones((BATCH_SIZE, 1), np.int32)
            _, loss_value, summary = sess.run((train, loss, summary_op),
                                    feed_dict={inputs: image_batch, labels: label_batch},
                                    options=run_options,
                                    run_metadata=run_metadata)
            print(loss_value)

            train_writer.add_run_metadata(run_metadata, 'step_%d' % i)
            train_writer.add_summary(summary, i)

if __name__ == '__main__':
    main()
