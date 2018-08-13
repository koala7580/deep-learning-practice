#pragma once
#ifndef __CUDNN_NDARRAY_H__
#define __CUDNN_NDARRAY_H__

#include "cudnn.h"

namespace cudnn {
    class Array4f32 {
        const size_t _dims[4];
        float* _data;

        inline size_t _index(size_t idx0, size_t idx1, size_t idx2, size_t idx3) const noexcept {
            return idx3 + idx2 * _dims[3]
                        + idx1 * _dims[2] * _dims[3]
                        + idx0 * _dims[1] * _dims[2] * _dims[3];
        }
    public:
        Array4f32(size_t dim0, size_t dim1, size_t dim2, size_t dim3);
        ~Array4f32();
        Array4f32(const Array4f32& other) = delete;
        Array4f32(Array4f32&& other);
        Array4f32& operator=(const Array4f32& other) = delete;
        Array4f32& operator=(Array4f32&& other) = delete;

        inline size_t dim(size_t dim) const noexcept { return _dims[dim]; };
        inline size_t size() const noexcept { return _dims[0] * _dims[1] * _dims[2] * _dims[3] * sizeof(float); }
        float* data() const noexcept { return _data; }
        float& operator()(size_t idx0, size_t idx1, size_t idx2, size_t idx3) noexcept;
        float operator()(size_t idx0, size_t idx1, size_t idx2, size_t idx3) const noexcept;
        Array4f32& operator=(float value) noexcept;

        void InitializeWithZeros();

        void Print() const noexcept;
    };
}
#endif // __CUDNN_NDARRAY_H__