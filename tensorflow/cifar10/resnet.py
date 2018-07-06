"""ResNet model
论文地址：https://arxiv.org/pdf/1512.03385.pdf
Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition.
    https://arxiv.org/pdf/1512.03385.pdf
The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks.
    https://arxiv.org/pdf/1603.05027.pdf

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from resnet_model import Model


def build_model(inputs, args, mode, params):
    resnet_size = 20
    num_blocks = (resnet_size - 2) // 6
    model = Model(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=10,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        final_size=64,
        resnet_version=2,
        data_format='channels_last',
        dtype=tf.float32
    )

    return model(inputs, mode == tf.estimator.ModeKeys.TRAIN)
