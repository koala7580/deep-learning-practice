#pragma once
#ifndef __CUDNN_CONTEXT_H__
#define __CUDNN_CONTEXT_H__

#include "cudnn.h"

namespace cudnn {
    class Context {
        cudnnHandle_t _handle;
    public:
        const cudnnHandle_t &handle;

    public:
        Context();
        ~Context();
        Context(const Context& other) = delete;
        Context(Context&& other) = delete;
        Context& operator=(const Context& other) = delete;
        Context& operator=(Context&& other) = delete;
    };
}

#endif // __CUDNN_CONTEXT_H__