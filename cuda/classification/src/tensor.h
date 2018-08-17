#pragma once
#ifndef __CUDNN_TENSOR_H__
#define __CUDNN_TENSOR_H__

#include "cudnn.h"
#include "data_type.h"
#include "tensor_format.h"
#include "array4d.h"

namespace cudnn {
    class Tensor {
        cudnnTensorDescriptor_t _descriptor;

        TensorFormat _format;
        DataType _data_type;
        uint32_t _batch_size, _n_channels, _height, _width;

        void *h_data;
        void *d_data;
    public:
        const TensorFormat &format;
        const DataType &data_type;
        const uint32_t &batch_size;
        const uint32_t &n_channels;
        const uint32_t &height;
        const uint32_t &width;

    public:
        Tensor(uint32_t batch_size, uint32_t n_channels, uint32_t height, uint32_t width,
                TensorFormat format = TensorFormat::ChannelsFirst,
                DataType data_type = DataType::Float32);
        ~Tensor();
        Tensor(const Tensor& other);
        Tensor(Tensor&& other);
        Tensor& operator=(const Tensor& other) = delete;
        Tensor& operator=(Tensor&& other) = delete;

        void SetShape(uint32_t batch_size, uint32_t n_channels, uint32_t height, uint32_t width);

        cudnnTensorDescriptor_t descriptor();
        inline size_t size() const {
            return _batch_size * _n_channels * height * width * size_of_data_type(_data_type);
        }
    };
}

#endif // __CUDNN_TENSOR_H__
