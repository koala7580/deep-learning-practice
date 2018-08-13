#pragma once
#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "cublas_v2.h"
#include "cudnn_context.h"
#include "tensor.h"

namespace layers {
    class Linear {
        const cudnn::Context &_context;
        cublasHandle_t _cublas;

        float* _weight_data;
        bool _use_bias;
        cudnn::Tensor4d _bias;
    public:
        size_t in_features, out_features;
        const size_t &n_rows;
        const size_t &n_cols;
        cudnn::Array4f32 bias_data;

    public:
        Linear(const cudnn::Context &context,
                int in_features,
                int out_features,
                bool use_bias=true);
        ~Linear();
        Linear(const Linear& other) = delete;
        Linear(Linear&& other) = delete;
        Linear& operator=(const Linear& other) = delete;
        Linear& operator=(Linear&& other) = delete;

        void Forward(const cudnn::Tensor4d &input_tensor,
                     const cudnn::Array4f32 &input_data,
                     const cudnn::Tensor4d &output_tensor,
                     cudnn::Array4f32 &output_data);

        inline size_t size() const noexcept { return n_rows * n_cols * sizeof(float); }
        inline float weight(size_t row, size_t col) const { return _weight_data[col + row * out_features]; }
        inline float& weight(size_t row, size_t col) { return _weight_data[col + row * out_features]; }
        void weight(float *data);

        cudnn::Tensor4d CreateOutputTensor(const cudnn::Tensor4d &input_tensor);
    };
}

#endif // __LINEAR_H__