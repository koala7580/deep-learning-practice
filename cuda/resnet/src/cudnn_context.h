#pragma once
#ifndef __CUDNN_CONTEXT_H__
#define __CUDNN_CONTEXT_H__

#include <cudnn.h>

namespace cudnn {
    class Context {
        cudnnHandle_t _handle;
    public:
        Context();
        ~Context();
        Context(const Context& other) = delete;
        Context(Context&& other) = delete;
        Context& operator=(const Context& other) = delete;
        Context& operator=(Context&& other) = delete;

        explicit operator cudnnHandle_t() const noexcept { return _handle; }
    };
}

#endif // __CUDNN_CONTEXT_H__