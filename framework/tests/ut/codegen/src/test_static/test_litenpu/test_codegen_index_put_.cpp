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

using namespace npu::tile_fwk;

class LiteNPUCodeGenIndexPut : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}
};

TEST_F(LiteNPUCodeGenIndexPut, test_index_put_001) {
    PROGRAM("INDEX_PUT_001") {
        Tensor self(DataType::DT_FP16, {3,3}, "self");
        Tensor values(DataType::DT_FP16, {2,3}, "values");
        Tensor indice(DataType::DT_FP16, {2}, "values");
        std::vector<Tensor> indices = {indice};
        bool accumulate = false;

        FUNCTION("INDEX_PUT_001") {           
            TileShape::Current().SetVecTile({3});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
