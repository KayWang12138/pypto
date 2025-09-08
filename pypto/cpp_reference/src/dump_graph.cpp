// Minimum graph dump
// Extracted from
// https://gitee.com/cann/ascendcpp/blob/3c2628cb051da5dac52332152732ab203fb675ad/tests/ut/operator/src/test_operation_impl.cpp

#include <iostream>

#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"

using namespace npu::tile_fwk;

int main() {
    // TODO: proper platform configuration to avoid run-time error
    Program::GetInstance().GetConfig().Reset();

    std::vector<int> shape{1, 2, 256, 128, 2};
    Tensor a(DT_FP32, shape, "a");
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 32, 32, 2);

    FUNCTION("BNSD2_BNS2D") {
        a = Transpose(a, {3, 4});
    }
    a->Dump();
    // std::cout << Program::GetInstance().Dump() << std::endl;
}
