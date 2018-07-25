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
"""Contains the core layers: PriorBox.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops


class PriorBox(base.Layer):
    """Generate the prior boxes of designated sizes and aspect ratios.
      Arguments
        img_size: Size of the input image as tuple (w, h).
        min_size: Minimum box size in pixels.
        max_size: Maximum box size in pixels.
        aspect_ratios: List of aspect ratios of boxes.
        flip: Whether to consider reverse aspect ratios.
        variances: List of variances for x, y, w, h.
        clip: Whether to clip the prior's coordinates
            such that they are within [0, 1].
      Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
      Output shape
        3D tensor with shape:
        (samples, num_boxes, 8)
      References
        https://arxiv.org/abs/1512.02325
     TODO
        Add possibility not to have variances.
        Add Theano support
    """

    def __init__(self,
                 img_size,
                 min_size,
                 max_size=None,
                 aspect_ratios=None,
                 flip=True,
                 variances=None,
                 clip=True,
                 data_format=None,
                 **kwargs):
        assert min_size > 0, 'min_size must be positive.'

        if variances is None:
            variances = [0.1]

        if data_format == 'channels_first':
            self.w_axis = 2
            self.h_axis = 1
        else:
            self.w_axis = 3
            self.h_axis = 2

        self.img_size = img_size
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]

        if max_size:
            assert max_size > min_size, \
                'max_size must be greater than min_size.'
            self.aspect_ratios.append(1.0)

        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue

                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)

        self.variances = np.array(variances)
        self.clip = clip
        super(PriorBox, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        num_priors = len(self.aspect_ratios)
        layer_width = input_shape[self.w_axis]
        layer_height = input_shape[self.h_axis]
        num_boxes = num_priors * layer_width * layer_height

        return input_shape[0], num_boxes, 8

    def call(self, inputs, mask=None):
        input_shape = inputs.shape

        layer_width = input_shape[self.w_axis]
        layer_height = input_shape[self.h_axis]
        img_height, img_width = self.img_size

        # define prior boxes shapes
        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # define centers of prior boxes
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        lin_x = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                            layer_width)
        lin_y = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                            layer_height)
        centers_x, centers_y = np.meshgrid(lin_x, lin_y)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # define xmin, ymin, xmax, ymax of prior boxes
        num_priors = len(self.aspect_ratios)
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)

        if self.clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)

        # define variances
        num_boxes = len(prior_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise RuntimeError('Must provide one or four variances.')

        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        prior_boxes_tensor = tf.expand_dims(K.variable(prior_boxes), 0)

        pattern = [tf.shape(x)[0], 1, 1]
        prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern)

        return prior_boxes_tensor
