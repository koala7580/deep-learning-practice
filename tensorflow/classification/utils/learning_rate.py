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
"""Learning rate utilities."""
import tensorflow as tf


class LearningRateWithDecay(object):

    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate


class PiecewiseLearningRate(LearningRateWithDecay):

    def __init__(self,
                 initial_learning_rate,
                 batches_per_epoch,
                 boundary_epochs,
                 decay_rates):
        super(PiecewiseLearningRate, self).__init__(initial_learning_rate)

        self.boundaries = [int(batches_per_epoch * epoch)
                           for epoch in boundary_epochs]
        self.values = [initial_learning_rate * decay
                       for decay in decay_rates]

    def __call__(self, global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step,
                                           self.boundaries,
                                           self.values)
