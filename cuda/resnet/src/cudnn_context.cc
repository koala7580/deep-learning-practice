#include "exception.h"
#include "cudnn_context.h"

cudnn::Context::Context() : handle(_handle)
{
    assert_cudnn_success( cudnnCreate(&_handle) );
}

cudnn::Context::~Context()
{
    assert_cudnn_success( cudnnDestroy(_handle) );
}
