#include "exception.h"
#include "cudnn_context.h"

cudnn::Context::Context()
{
    assert_cudnn_success( cudnnCreate(&_handle) );
}

cudnn::Context::~Context()
{
    cudnnDestroy(_handle);
}
