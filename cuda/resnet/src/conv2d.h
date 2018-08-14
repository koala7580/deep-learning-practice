#pragma once
#ifndef __CONV1D_H__
#define __CONV1D_H__

#include "cudnn_context.h"
#include "tensor.h"
#include "kernel.h"

namespace layers {
    class Conv1D {
        const cudnn::Context &_context;
        const cudnn::Kernel _kernel;
        const cudnn::Tensor4d _bias;
        int _stride;
        int _padding;
        bool _use_bias;

        cudnnConvolutionDescriptor_t _convolution_descriptor;
        cudnnConvolutionFwdAlgo_t _convolution_fwd_algo;
        void *_workspace;
        size_t _workspace_size;

        void _PrepareWorkspace(const cudnn::Tensor4d &input_tensor,
                               const cudnn::Tensor4d &output_tensor);
    public:
        size_t in_channels;
        size_t out_channels;
        size_t kernel_size;
        cudnn::Array4f32 weight_data;
        cudnn::Array4f32 bias_data;

    public:
        Conv1D(const cudnn::Context &context,
                int in_channels,
                int out_channels,
                int kernel_size,
                int stride=1,
                int padding=0,
                bool use_bias=true);
        ~Conv1D();
        Conv1D(const Conv1D& other) = delete;
        Conv1D(Conv1D&& other) = delete;
        Conv1D& operator=(const Conv1D& other) = delete;
        Conv1D& operator=(Conv1D&& other) = delete;

        void Forward(const cudnn::Tensor4d &input_tensor,
                     const cudnn::Array4f32 &input_data,
                     const cudnn::Tensor4d &output_tensor,
                     cudnn::Array4f32 &output_data);

        cudnn::Tensor4d CreateOutputTensor(const cudnn::Tensor4d &input_tensor);
    };
}

#endif // __CONV1D_H__