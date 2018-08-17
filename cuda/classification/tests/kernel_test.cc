#include "catch.hpp"

#include "kernel.h"

TEST_CASE( "Kernel basic test.", "[cudnn:Kernel]" ) {
    SECTION("constructor") {
        cudnn::Kernel k(3, 3, 3, 3);
        REQUIRE_NOTHROW( k.descriptor() != nullptr );
    }
}
