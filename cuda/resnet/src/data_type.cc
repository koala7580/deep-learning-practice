#include <iostream>
#include <stdexcept>
#include "data_type.h"

size_t cudnn::size_of_data_type(cudnn::DataType dataType) {
    switch(dataType) {
        case DataType::Float16: return sizeof(float);
        case DataType::Float32: return sizeof(float);
        case DataType::Float64: return sizeof(double);
        // case DataType::Int8: return sizeof(int8_t);
        // case DataType::UInt8: return sizeof(uint8_t);
        case DataType::Int32: return sizeof(uint32_t);
        default:
            throw std::invalid_argument("The given DataType is invalid.");
    }
}

template<>
bool cudnn::is_valid_type<float>(DataType dt) {
    return dt == DataType::Float16 || dt == DataType::Float32;
}

template<>
bool cudnn::is_valid_type<double>(DataType dt) {
    return dt == DataType::Float64;
}
