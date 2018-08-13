#include <utility>
#include <iostream>
#include "tensor.h"
#include "exception.h"

cudnn::Tensor4d::Tensor4d(size_t batch_size, size_t n_channels, size_t height, size_t width,
        TensorFormat format, DataType data_type)
: batch_size(batch_size), n_channels(n_channels), height(height), width(width),
  format(format), data_type(data_type),
  _descriptor(nullptr)
{
    assert_cudnn_success( cudnnCreateTensorDescriptor(&_descriptor) );
    assert_cudnn_success( cudnnSetTensor4dDescriptor(_descriptor,
        static_cast<cudnnTensorFormat_t>(format),
        static_cast<cudnnDataType_t>(data_type),
        batch_size, n_channels, height, width
    ) );
}

cudnn::Tensor4d::Tensor4d(const cudnn::Tensor4d &other)
: batch_size(other.batch_size), n_channels(other.n_channels), height(other.height), width(other.width),
  format(other.format), data_type(other.data_type)
{
    assert_cudnn_success( cudnnCreateTensorDescriptor(&_descriptor) );
    assert_cudnn_success( cudnnSetTensor4dDescriptor(_descriptor,
        static_cast<cudnnTensorFormat_t>(format),
        static_cast<cudnnDataType_t>(data_type),
        batch_size, n_channels, height, width
    ) );
}

cudnn::Tensor4d::Tensor4d(cudnn::Tensor4d &&other)
: batch_size(other.batch_size), n_channels(other.n_channels), height(other.height), width(other.width),
  format(other.format), data_type(other.data_type),
  _descriptor(std::exchange(other._descriptor, nullptr))
{
    
}

cudnn::Tensor4d::~Tensor4d()
{
    if (_descriptor != nullptr) {
        assert_cudnn_success( cudnnDestroyTensorDescriptor(_descriptor) );
    }
}

cudnn::Array4f32 cudnn::Tensor4d::CreateArray4f32() const
{
    switch(format) {
        case TensorFormat::ChannelsFirst:
            return Array4f32(batch_size, n_channels, height, width);
        case TensorFormat::ChannelsLast:
            return Array4f32(batch_size, height, width, n_channels);
        default:
            throw std::runtime_error("Unknown tensor format.");
    }
}
