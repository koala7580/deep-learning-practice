#include <iostream>
#include "linear.h"
#include "exception.h"

layers::Linear::Linear(
    const cudnn::Context &context,
    int in_features,
    int out_features,
    bool use_bias)
: _context(context),
  _weight_data(nullptr),
  _use_bias(use_bias),
  _bias(1, out_features, 1, 1),

  in_features(in_features),
  out_features(out_features),
  n_rows(this->in_features),
  n_cols(this->out_features),
  bias_data(_bias.CreateArray4f32())
{
    assert_cublas_success( cublasCreate(&_cublas) );

    size_t size = n_rows * n_cols * sizeof(float);
    assert_cuda_success( cudaMallocManaged(&_weight_data, size) );
    assert_cuda_success( cudaMemset(_weight_data, 0, size) );
    assert_cuda_success( cudaDeviceSynchronize() );
}

layers::Linear::~Linear()
{
    assert_cublas_success( cublasDestroy(_cublas) );
    assert_cuda_success( cudaFree(_weight_data) );
}

cudnn::Tensor4d
layers::Linear::CreateOutputTensor(const cudnn::Tensor4d &input_tensor)
{
    return cudnn::Tensor4d(input_tensor.batch_size, out_features,
                           input_tensor.height, input_tensor.width);
}

void layers::Linear::Forward(
    const cudnn::Tensor4d &input_tensor,
    const cudnn::Array4f32 &input_data,
    const cudnn::Tensor4d &output_tensor,
    cudnn::Array4f32 &output_data)
{
    float *x = input_data.data();
    float *y = output_data.data();
    const float alpha = 1.0, beta = 0.0;
    for(int batch = 0; batch < input_tensor.batch_size; batch++) {
        for(int h = 0; h < input_tensor.height; h++) {
            for(int w = 0; w < input_tensor.width; w ++) {
                size_t width = input_tensor.width;
                size_t height = input_tensor.height;
                float *xx = x + w + h * width + batch * in_features * height * width;
                float *yy = y + w + h * width + batch * out_features * height * width;
                assert_cublas_success(
                    cublasSgemv_v2(_cublas, CUBLAS_OP_T,
                                   n_rows, n_cols, &alpha, _weight_data, n_rows,
                                   xx, width * height, &beta,
                                   yy, width * height) );
            }
        }
    }

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

void layers::Linear::weight(float *data)
{
    memcpy(_weight_data, data, this->size());
}
