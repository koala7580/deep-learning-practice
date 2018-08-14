#pragma once
#ifndef __CUDNN_KERNEL_H__
#define __CUDNN_KERNEL_H__

#include <cassert>
#include "cudnn.h"
#include "data_type.h"
#include "tensor_format.h"
#include "array4d.h"

namespace cudnn {
    class Kernel {
        cudnnFilterDescriptor_t _descriptor;

    public:
        const TensorFormat format;
        const DataType data_type;
        const size_t out_channels;
        const size_t in_channels;
        const size_t height;
        const size_t width;

    public:
        Kernel(int in_channels, int out_channels, int height, int width,
                TensorFormat format = TensorFormat::ChannelsFirst,
                DataType dataType = DataType::Float32);
        ~Kernel();

        Kernel(const Kernel& other) = delete;
        Kernel(Kernel&& other) = delete;
        Kernel& operator=(const Kernel& other)  = delete;
        Kernel& operator=(Kernel&& other) = delete;

        inline cudnnFilterDescriptor_t descriptor() const noexcept { return _descriptor; }
    };
}

#endif // __CUDNN_KERNEL_H__
