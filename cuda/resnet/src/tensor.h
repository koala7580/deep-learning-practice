#pragma once
#ifndef __CUDNN_TENSOR_H__
#define __CUDNN_TENSOR_H__

#include "cudnn.h"
#include "data_type.h"
#include "tensor_format.h"
#include "array4d.h"

namespace cudnn {
    class Tensor4d {
        cudnnTensorDescriptor_t _descriptor;
    public:
        const TensorFormat format;
        const DataType data_type;
        const int batch_size, n_channels, height, width;

    public:
        Tensor4d(size_t batch_size, size_t n_channels, size_t height, size_t width,
                TensorFormat format = TensorFormat::ChannelsFirst,
                DataType data_type = DataType::Float32);
        ~Tensor4d();
        Tensor4d(const Tensor4d& other);
        Tensor4d(Tensor4d&& other);
        Tensor4d& operator=(const Tensor4d& other) = delete;
        Tensor4d& operator=(Tensor4d&& other) = delete;

        Array4f32 CreateArray4f32() const;

        explicit operator cudnnTensorDescriptor_t() const noexcept { return _descriptor; }
    };
}

#endif // __CUDNN_TENSOR_H__
