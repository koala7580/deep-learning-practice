#include "catch.hpp"

#include "cudnn_context.h"

TEST_CASE( "cuDNN Context Wrapper", "[cudnn::Context]" ) {
    cudnn::Context context;
    
    SECTION( "convert to cudnnContext_t" ) {
        REQUIRE( context.handle != nullptr );
    }
}
