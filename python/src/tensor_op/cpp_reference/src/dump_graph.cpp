/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dump_graph.cpp
 * \brief Minimum graph dump
 */

#include <iostream>

#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"

using namespace npu::tile_fwk;

int main() {
    Program::GetInstance().GetConfig().Reset();

    std::vector<int> shape{1, 2, 256, 128, 2};
    Tensor a(DT_FP32, shape, "a");
    TileShape::Current().SetVecTile(1, 1, 32, 32, 2);

    FUNCTION("BNSD2_BNS2D") {
        a = Transpose(a, {3, 4});
    }
    a->Dump();
    // std::cout << Program::GetInstance().Dump() << std::endl;
}
