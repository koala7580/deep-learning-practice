#include <iostream>
#include "kernel.h"
#include "exception.h"

cudnn::Kernel::Kernel(int in_channels, int out_channels, int height, int width,
                     TensorFormat format, DataType data_type)
: out_channels(out_channels), in_channels(in_channels),
  height(height), width(width), format(format), data_type(data_type)
{
    assert_cudnn_success( cudnnCreateFilterDescriptor(&_descriptor) );
    assert_cudnn_success( cudnnSetFilter4dDescriptor(_descriptor,
        static_cast<cudnnDataType_t>(data_type),
        static_cast<cudnnTensorFormat_t>(format),
        out_channels,
        in_channels,
        height,
        width
    ));
}

cudnn::Kernel::~Kernel()
{
    assert_cudnn_success( cudnnDestroyFilterDescriptor(_descriptor) );
}

