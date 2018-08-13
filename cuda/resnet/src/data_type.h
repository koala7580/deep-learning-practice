#pragma once
#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include "cudnn.h"

namespace cudnn {

    enum class DataType {
        Float16 = CUDNN_DATA_HALF,
        Float32 = CUDNN_DATA_FLOAT,
        Float64 = CUDNN_DATA_DOUBLE,
        // Int8 = CUDNN_DATA_INT8,
        // UInt8 = CUDNN_DATA_UINT8,
        Int32 = CUDNN_DATA_INT32,
        Invalid
    };

    size_t size_of_data_type(DataType dataType);

    template<typename T>
    bool is_valid_type(DataType dt) { return false; }
    template<> bool is_valid_type<float>(DataType);
    template<> bool is_valid_type<double>(DataType);
}

#endif // __DATA_TYPE_H__