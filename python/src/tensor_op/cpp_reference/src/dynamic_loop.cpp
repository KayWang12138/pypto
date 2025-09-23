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
 * \file dynamic_loop.cpp
 * \brief
 */

#include <iostream>

#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

int main(){
    std::vector<int> shape = {128, 128};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    FUNCTION("main", FunctionType::DYNAMIC, {a}, {b}) {
        TileShape::Current().SetVecTile({64, 64});
        LOOP("Dynamic", FunctionType::DYNAMIC_LOOP, k, LoopRange(10)) {
            b = Add(a, a); // 运算1
            IF (k < 2) {
                b = Add(b, a); // 运算2
            } ELSE {
                b = Sub(b, a); // 运算3
            }

            IF (k < 5) {
                b = Mul(b, a); // 运算4
            } ELSE {
                b = Div(b, a); // 运算5
            }
            b = Sub(b, a); // 运算6
        }
    }
    std::cout << "finished" << std::endl;
}
