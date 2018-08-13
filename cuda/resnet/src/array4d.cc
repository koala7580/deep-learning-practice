#include <cassert>
#include <utility>
#include <cstring>
#include <iostream>
#include "array4d.h"
#include "exception.h"

cudnn::Array4f32::Array4f32(size_t dim0, size_t dim1, size_t dim2, size_t dim3)
: _dims { dim0, dim1, dim2, dim3 }, _data{nullptr}
{
    assert_cuda_success( cudaMallocManaged(&_data, this->size()) );
}

cudnn::Array4f32::Array4f32(Array4f32&& other)
: _dims{ other._dims[0], other._dims[1], other._dims[2], other._dims[3] },
  _data{std::exchange(other._data, nullptr)}
{

}

cudnn::Array4f32::~Array4f32()
{
    if (_data != nullptr) {
        assert_cuda_success( cudaFree(_data) );
    }
}

float& cudnn::Array4f32::operator()(size_t idx1, size_t idx2, size_t idx3, size_t idx4) noexcept
{
    assert(idx1 <= _dims[0]);
    assert(idx2 <= _dims[1]);
    assert(idx3 <= _dims[2]);
    assert(idx4 <= _dims[3]);

    size_t index = this->_index(idx1, idx2, idx3, idx4);
    return _data[index];
}

float cudnn::Array4f32::operator()(size_t idx1, size_t idx2, size_t idx3, size_t idx4) const noexcept
{
    return const_cast<Array4f32&>(*this)(idx1, idx2, idx3, idx4);
}

void cudnn::Array4f32::InitializeWithZeros()
{
    size_t size = this->size();
    assert_cuda_success( cudaMemset(_data, 0, size) );
}

cudnn::Array4f32& cudnn::Array4f32::operator=(float value) noexcept
{
    for (size_t idx0 = 0; idx0 < _dims[0]; ++idx0) {
        for (size_t idx1 = 0; idx1 < _dims[1]; ++idx1) {
            for (size_t idx2 = 0; idx2 < _dims[2]; ++idx2) {
                for (size_t idx3 = 0; idx3 < _dims[3]; ++idx3) {
                    _data[this->_index(idx0, idx1, idx2, idx3)] = value;
                }
            }
        }
    }

    return *this;
}

void cudnn::Array4f32::Print() const noexcept
{
    for (size_t idx0 = 0; idx0 < _dims[0]; ++idx0) {
        for (size_t idx1 = 0; idx1 < _dims[1]; ++idx1) {
            std::cout << "[" << idx0 << ", " << idx1 << "]" << std::endl;
            for (size_t idx2 = 0; idx2 < _dims[2]; ++idx2) {
                for (size_t idx3 = 0; idx3 < _dims[3]; ++idx3) {
                    std::cout << _data[this->_index(idx0, idx1, idx2, idx3)];
                    std::cout << (idx3 + 1 < _dims[3] ? ", " : "");
                }
                std::cout << std::endl;
            }
        }
    }
}
