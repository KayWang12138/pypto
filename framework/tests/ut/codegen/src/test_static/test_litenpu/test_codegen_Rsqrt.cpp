/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_operation_impl.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "interface/interpreter/calc.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/litenpu/codegen_litenpu.h"
using namespace npu::tile_fwk;


class LiteNPUCodegenRsqrt : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    }

    void TearDown() override {}
};


TEST_F(LiteNPUCodegenRsqrt, test_Rsqrt_lite_002) {
    PROGRAM("RSQRT") {
        Tensor input(DataType::DT_FP16, {100}, "input");
        auto output = Tensor(DataType::DT_FP16, {100}, "output");
        FUNCTION("RSQRT_002") {           
        TileShape::Current().SetVecTile({100});
            output = Rsqrt(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RSQRT_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}


