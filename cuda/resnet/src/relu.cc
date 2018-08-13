#include "relu.h"
#include "exception.h"

cudnn::ReLU::ReLU(const cudnn::Context &context)
: _context(context)
{
    assert_cudnn_success( cudnnCreateActivationDescriptor(&_activation_descriptor) );
    assert_cudnn_success( cudnnSetActivationDescriptor(_activation_descriptor,
        CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0
    ) );
}

cudnn::ReLU::~ReLU()
{
    assert_cudnn_success( cudnnDestroyActivationDescriptor(_activation_descriptor) );
}

void cudnn::ReLU::operator()(
    const Tensor4d &input_tensor,
    const Array4f32 &input_data,
    Array4f32 &output_data )
{
    const float alpha = 1.0, beta = 0.0;
    assert_cudnn_success( cudnnActivationForward(
        static_cast<cudnnHandle_t>(_context),
        _activation_descriptor,
        &alpha,
        static_cast<cudnnTensorDescriptor_t>(input_tensor),
        input_data.data(),
        &beta,
        static_cast<cudnnTensorDescriptor_t>(input_tensor),
        output_data.data()
    ) );
}
