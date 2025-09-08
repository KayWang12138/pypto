
#include <iostream>

#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"

using namespace npu::tile_fwk;

int main() {
    auto config = Program::GetInstance().GetConfig();
    config.Reset();
    config.Set("ONLY_CODEGEN", true);

    // TODO: allow configuring UB size to allow larger tile sizes
    std::vector<int> shape = {128, 2, 64, 128};
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape,  "B");
    Tensor output(DT_FP32, shape, "C");
    FUNCTION("ADD") {
        TileShape::Current().SetVecTile({32, 1, 16, 32});
        // NOTE: here increase TileShapes from {32, 1, 16, 32} to {32, 1, 64, 32} will 
        // crash `OOOSchedule` pass due to UB overflow
        output = Add(input_a, input_b);
    } // EndFunction

    output->Dump();
    std::cout << Program::GetInstance().Dump() << std::endl;
}
