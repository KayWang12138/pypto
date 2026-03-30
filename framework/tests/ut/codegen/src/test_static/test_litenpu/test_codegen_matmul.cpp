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
 * \file test_codegen_matmul.cpp
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
#include "codegen/litenpu/codegen_litenpu.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

using namespace npu::tile_fwk;

class LiteNPUCodeGenMatmul : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetBuildStatic(true);
    }

    void TearDown() override {}
};

TEST_F(LiteNPUCodeGenMatmul, test_matmul_001) {
    PROGRAM("MATMUL_001") {
        Tensor a(DataType::DT_FP16, {16, 16}, "a");
        Tensor b(DataType::DT_FP16, {16, 16}, "b");
        auto c = Tensor(DataType::DT_FP16, {16, 16}, "c");
        FUNCTION("MATMUL_001") {
            TileShape::Current().SetCubeTile({16, 16}, {16, 16}, {16, 16}, false, false);
            c = npu::tile_fwk::Matrix::Matmul(DataType::DT_FP16, a, b, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenMatmul, test_matmul_001_cloud_test) {
    PROGRAM("MATMUL_001_CLOUD") {
        Tensor a(DataType::DT_FP16, {16, 16}, "a");
        Tensor b(DataType::DT_FP16, {16, 16}, "b");
        auto c = Tensor(DataType::DT_FP16, {16, 16}, "c");
        FUNCTION("MATMUL_001_CLOUD") {
            TileShape::Current().SetCubeTile({16, 16}, {16, 16}, {16, 16}, false, false);
            c = npu::tile_fwk::Matrix::Matmul(DataType::DT_FP16, a, b, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_001_CLOUD");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
