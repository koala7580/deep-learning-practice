#pragma once
#ifndef __RELU_H__
#define __RELU_H__

#include "cudnn_context.h"
#include "tensor.h"

namespace cudnn {
    class ReLU {
        const Context &_context;
        cudnnActivationDescriptor_t _activation_descriptor;
    public:
        ReLU(const Context &context);
        ~ReLU();
        ReLU(const ReLU& other) = delete;
        ReLU(ReLU&& other) = delete;
        ReLU& operator=(const ReLU& other) = delete;
        ReLU& operator=(ReLU&& other) = delete;

        void operator()(const Tensor4d &input_tensor, const Array4f32 &input_data, Array4f32 &output_data);
    };
}

#endif // __RELU_H__