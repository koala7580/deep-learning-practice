#include "conv1d.h"
#include "exception.h"

layers::Conv1D::Conv1D(
    const cudnn::Context &context,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    bool use_bias)
: _context(context),
  _kernel(in_channels, out_channels, 1, kernel_size),
  _bias(1, out_channels, 1, 1),
  _stride(stride),
  _padding(padding),
  _use_bias(use_bias),
  _workspace(nullptr),
  _workspace_size(0),
  in_channels(in_channels),
  out_channels(out_channels),
  kernel_size(kernel_size),
  weight_data(_kernel.CreateArray4f32()),
  bias_data(_bias.CreateArray4f32())
{
    assert_cudnn_success( cudnnCreateConvolutionDescriptor(&_convolution_descriptor) );
    assert_cudnn_success( cudnnSetConvolution2dDescriptor(_convolution_descriptor,
        0, padding, 1, stride, 1, 1,
        CUDNN_CROSS_CORRELATION,
        static_cast<cudnnDataType_t>(_kernel.format)
    ) );
}

layers::Conv1D::~Conv1D()
{
    assert_cudnn_success( cudnnDestroyConvolutionDescriptor(_convolution_descriptor) );
    if (_workspace != nullptr) {
        assert_cuda_success( cudaFree(_workspace) );
    }
}

cudnn::Tensor4d layers::Conv1D::CreateOutputTensor(
    const cudnn::Tensor4d &input_tensor)
{
    int l_out = (input_tensor.width + 2 * this->_padding - _kernel.width) / this->_stride + 1;
    return cudnn::Tensor4d(input_tensor.batch_size, _kernel.out_channels, 1, l_out);
}

void layers::Conv1D::Forward(
    const cudnn::Tensor4d &input_tensor,
    const cudnn::Array4f32 &input_data,
    const cudnn::Tensor4d &output_tensor,
    cudnn::Array4f32 &output_data)
{
    _PrepareWorkspace(input_tensor, output_tensor);

    const float alpha = 1.0, beta = 0.0;
    assert_cudnn_success( cudnnConvolutionForward(
        static_cast<cudnnHandle_t>(_context),
        &alpha,
        static_cast<cudnnTensorDescriptor_t>(input_tensor),
        input_data.data(),
        static_cast<cudnnFilterDescriptor_t>(_kernel),
        weight_data.data(),
        static_cast<cudnnConvolutionDescriptor_t>(_convolution_descriptor),
        _convolution_fwd_algo,
        _workspace,
        _workspace_size,
        &beta,
        static_cast<cudnnTensorDescriptor_t>(output_tensor),
        output_data.data()) );

    if (_use_bias) {
        const float alpha = 1.0, beta = 1.0;
        assert_cudnn_success( cudnnAddTensor(
            static_cast<cudnnHandle_t>(_context),
            &alpha,
            static_cast<cudnnTensorDescriptor_t>(_bias),
            bias_data.data(),
            &beta,
            static_cast<cudnnTensorDescriptor_t>(output_tensor),
            output_data.data()) );
    }
}

void layers::Conv1D::_PrepareWorkspace(
    const cudnn::Tensor4d &input_tensor,
    const cudnn::Tensor4d &output_tensor)
{
    if (_workspace == nullptr) {
        assert_cudnn_success(cudnnGetConvolutionForwardAlgorithm(
            static_cast<cudnnHandle_t>(_context),
            static_cast<cudnnTensorDescriptor_t>(input_tensor),
            static_cast<cudnnFilterDescriptor_t>(_kernel),
            static_cast<cudnnConvolutionDescriptor_t>(_convolution_descriptor),
            static_cast<cudnnTensorDescriptor_t>(output_tensor),
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            /*memoryLimitInBytes=*/0,
            &_convolution_fwd_algo));

    }

    size_t workspace_size = 0;
    assert_cudnn_success(cudnnGetConvolutionForwardWorkspaceSize(
        static_cast<cudnnHandle_t>(_context),
        static_cast<cudnnTensorDescriptor_t>(input_tensor),
        static_cast<cudnnFilterDescriptor_t>(_kernel),
        static_cast<cudnnConvolutionDescriptor_t>(_convolution_descriptor),
        static_cast<cudnnTensorDescriptor_t>(output_tensor),
        _convolution_fwd_algo,
        &workspace_size));

    if (workspace_size != _workspace_size) {
        if (_workspace != nullptr) {
            assert_cuda_success( cudaFree(_workspace) );
        }

        _workspace_size = workspace_size;
        assert_cuda_success( cudaMalloc(&_workspace, _workspace_size) );
    }
}
