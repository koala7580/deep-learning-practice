#include "catch.hpp"

#include "tensor.h"

TEST_CASE( "Tensor4d is works", "[cudnn:Tensor4d]" ) {
    SECTION("constructor") {
        cudnn::Tensor t(3, 3, 3, 3);
        REQUIRE_NOTHROW(t.descriptor() != nullptr);
        REQUIRE(t.batch_size == 3);
    }
    SECTION("SetShape") {
        cudnn::Tensor t(3, 3, 3, 3);
        t.SetShape(1, 3, 3, 3);
        REQUIRE_NOTHROW(t.descriptor() != nullptr);
    }
}
