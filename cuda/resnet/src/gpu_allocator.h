#pragma once
#ifndef __GPU_ALLOCATOR_H__
#define __GPU_ALLOCATOR_H__

#include <list>

class GPUAllocator {
public:
    GPUAllocator();
    ~GPUAllocator();
    GPUAllocator(const GPUAllocator& other) = delete;
    GPUAllocator(GPUAllocator&& other) = delete;
    GPUAllocator& operator=(const GPUAllocator& other) = delete;
    GPUAllocator& operator=(GPUAllocator&& other) = delete;
};

#endif // __GPU_ALLOCATOR_H__