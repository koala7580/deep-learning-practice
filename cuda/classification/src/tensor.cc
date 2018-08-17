#include <utility>
#include <iostream>
#include "tensor.h"
#include "exception.h"

cudnn::Tensor::Tensor(uint32_t batch_size, uint32_t n_channels, uint32_t height, uint32_t width,
    cudnn::TensorFormat format, cudnn::DataType data_type)
:  _descriptor(nullptr),
  _batch_size(batch_size), _n_channels(n_channels), _height(height), _width(width),
  _format(format), _data_type(data_type),
  batch_size(this->_batch_size), n_channels(this->_n_channels),
  height(this->_height), width(this->_width),
  format(this->_format), data_type(this->_data_type)
{

}

cudnn::Tensor::Tensor(const cudnn::Tensor &other)
: _descriptor(nullptr),
  _batch_size(other.batch_size), _n_channels(other.n_channels), _height(other.height), _width(other.width),
  _format(other.format), _data_type(other.data_type),
  batch_size(this->_batch_size), n_channels(this->_n_channels),
  height(this->_height), width(this->_width),
  format(this->_format), data_type(this->_data_type)
{

}

cudnn::Tensor::Tensor(cudnn::Tensor &&other)
: _batch_size(other.batch_size), _n_channels(other.n_channels), _height(other.height), _width(other.width),
  _format(other.format), _data_type(other.data_type),
  batch_size(this->_batch_size), n_channels(this->_n_channels),
  height(this->_height), width(this->_width),
  format(this->_format), data_type(this->_data_type),
  _descriptor(std::exchange(other._descriptor, nullptr))
{
    //
}

cudnn::Tensor::~Tensor()
{
    if (_descriptor != nullptr) {
        assert_cudnn_success( cudnnDestroyTensorDescriptor(_descriptor) );
    }
}

void cudnn::Tensor::SetShape(uint32_t batch_size, uint32_t n_channels, uint32_t height, uint32_t width)
{
    _batch_size = batch_size;
    _n_channels = n_channels;
    _height = height;
    _width = width;
    
    if (_descriptor != nullptr) {
        assert_cudnn_success( cudnnSetTensor4dDescriptor(_descriptor,
            static_cast<cudnnTensorFormat_t>(_format),
            static_cast<cudnnDataType_t>(_data_type),
            _batch_size, _n_channels, _height, _width
        ));
    }
}

cudnnTensorDescriptor_t cudnn::Tensor::descriptor()
{
    if (_descriptor == nullptr) {
        assert_cudnn_success( cudnnCreateTensorDescriptor(&_descriptor) );
        assert_cudnn_success( cudnnSetTensor4dDescriptor(_descriptor,
            static_cast<cudnnTensorFormat_t>(_format),
            static_cast<cudnnDataType_t>(_data_type),
            _batch_size, _n_channels, _height, _width
        ) );
    }
    return _descriptor;
}
