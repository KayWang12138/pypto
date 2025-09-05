/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Similar to vector_add.cpp but using manual recording without MACROs

#include <iostream>

#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"

using namespace npu::tile_fwk;

int main() {
    std::vector<int> shape = {128, 2, 64, 128};
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape,  "B");
    Tensor output(DT_FP32, shape, "C");

    Program::GetInstance().BeginFunction("ADD");

    Program::GetInstance().GetTileShape().SetVecTileShapes({32, 1, 16, 32});
    output = Add(input_a, input_b);

    Program::GetInstance().EndFunction("ADD");

    output->Dump();
    std::cout << Program::GetInstance().Dump() << std::endl;
}
