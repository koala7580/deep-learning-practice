#include <cstdio>
#include <cstring>
#include "cudnn.h"
#include "exception.h"

char cudnn::Exception::_buffer[4096] = { 0 };
char cuda::Exception::_buffer[4096] = { 0 };
char cublas::Exception::_buffer[4096] = { 0 };

cudnn::Exception&
cudnn::Exception::operator=(const cudnn::Exception &other)
{
    return *this = Exception(other);    
}

cudnn::Exception&
cudnn::Exception::operator=(cudnn::Exception &&other) noexcept
{
    this->_status = other._status;
    this->_file = other._file;
    this->_line = other._line;
    return *this;
}

const char* cudnn::Exception::what() const noexcept {
    sprintf(this->_buffer, "Error in %s line %lu: %s",
            this->_file, this->_line, cudnnGetErrorString(this->_status));
    return this->_buffer;
}


cuda::Exception&
cuda::Exception::operator=(const cuda::Exception &other)
{
    return *this = Exception(other);    
}

cuda::Exception&
cuda::Exception::operator=(cuda::Exception &&other) noexcept
{
    this->_status = other._status;
    this->_file = other._file;
    this->_line = other._line;
    return *this;
}

const char* cuda::Exception::what() const noexcept {
    sprintf(this->_buffer, "Error in %s line %lu: %s",
            this->_file, this->_line, cudaGetErrorString(this->_status));
    return this->_buffer;
}

const char* cublas::Exception::what() const noexcept {
    const char *status_str = "<unknown>";
    switch(_status) {
        case CUBLAS_STATUS_SUCCESS: status_str = "CUBLAS_STATUS_SUCCESS"; break;
        case CUBLAS_STATUS_NOT_INITIALIZED: status_str = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
        case CUBLAS_STATUS_ALLOC_FAILED: status_str = "CUBLAS_STATUS_ALLOC_FAILED"; break;
        case CUBLAS_STATUS_INVALID_VALUE: status_str = "CUBLAS_STATUS_INVALID_VALUE"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH: status_str = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
        case CUBLAS_STATUS_MAPPING_ERROR: status_str = "CUBLAS_STATUS_MAPPING_ERROR"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED: status_str = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
        case CUBLAS_STATUS_INTERNAL_ERROR: status_str = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
        case CUBLAS_STATUS_NOT_SUPPORTED: status_str = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
        case CUBLAS_STATUS_LICENSE_ERROR: status_str = "CUBLAS_STATUS_LICENSE_ERROR"; break;
        default:
            status_str = "<unknown>";
    }

    sprintf(this->_buffer, "Error in %s line %lu: %s",
            this->_file, this->_line, status_str);
    return this->_buffer;
}
