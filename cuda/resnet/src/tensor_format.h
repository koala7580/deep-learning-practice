#pragma once
#ifndef __CUDNN_TENSOR_FORMAT_H__
#define __CUDNN_TENSOR_FORMAT_H__

#include "cudnn.h"

namespace cudnn {
    enum class TensorFormat {
        ChannelsFirst = CUDNN_TENSOR_NCHW,
        ChannelsLast = CUDNN_TENSOR_NHWC
    };
}
#endif // __CUDNN_TENSOR_FORMAT_H__