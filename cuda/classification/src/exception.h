#pragma once
#ifndef __EXCEPTION_H__
#define __EXCEPTION_H__

#include <exception>
#include "cudnn.h"
#include "cublas_v2.h"

#define assert_cudnn_success(expression)                     \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
        throw cudnn::Exception(status, __FILE__, __LINE__);  \
    }                                                        \
  }

#define assert_cuda_success(expression)                     \
  {                                                         \
    cudaError_t status = (expression);                      \
    if (status != cudaSuccess) {                            \
        throw cuda::Exception(status, __FILE__, __LINE__);  \
    }                                                       \
  }

#define assert_cublas_success(expression)                     \
  {                                                           \
    cublasStatus_t status = (expression);                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
        throw cublas::Exception(status, __FILE__, __LINE__);  \
    }                                                         \
  }

namespace cudnn {

    class Exception : public std::exception {
        static char _buffer[1024 * 4];

        size_t _line;
        const char * _file;
        cudnnStatus_t _status;

    public:
        Exception(cudnnStatus_t status, const char *file, size_t line) noexcept
        : _status(status), _file(file), _line(line)
        {}
        ~Exception() override {}

        Exception(const Exception& other) noexcept // copy constructor
        : _status(other._status), _file(other._file), _line(other._line)
        {}

        Exception(Exception&& other) noexcept // move constructor
        : _status(other._status), _file(other._file), _line(other._line)
        {}

        Exception& operator=(const Exception& other); // copy assignment
        Exception& operator=(Exception&& other) noexcept; // move assignment

        const char* what() const noexcept override;
    };
}

namespace cuda {

    class Exception : public std::exception {
        static char _buffer[1024 * 4];

        size_t _line;
        const char * _file;
        cudaError_t _status;

    public:
        Exception(cudaError_t status, const char *file, size_t line) noexcept
        : _status(status), _file(file), _line(line)
        {}
        ~Exception() override {}

        Exception(const Exception& other) noexcept // copy constructor
        : _status(other._status), _file(other._file), _line(other._line)
        {}

        Exception(Exception&& other) noexcept // move constructor
        : _status(other._status), _file(other._file), _line(other._line)
        {}

        Exception& operator=(const Exception& other); // copy assignment
        Exception& operator=(Exception&& other) noexcept; // move assignment

        const char* what() const noexcept override;
    };
}

namespace cublas {

    class Exception : public std::exception {
        static char _buffer[1024 * 4];

        size_t _line;
        const char * _file;
        cublasStatus_t _status;

    public:
        Exception(cublasStatus_t status, const char *file, size_t line) noexcept
        : _status(status), _file(file), _line(line)
        {}
        ~Exception() override {}

        Exception(const Exception& other) noexcept // copy constructor
        : _status(other._status), _file(other._file), _line(other._line)
        {}

        Exception(Exception&& other) noexcept // move constructor
        : _status(other._status), _file(other._file), _line(other._line)
        {}

        Exception& operator=(const Exception& other); // copy assignment
        Exception& operator=(Exception&& other) noexcept; // move assignment

        const char* what() const noexcept override;
    };
}

#endif // __EXCEPTION_H__