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
 * \file test_codegen_full.cpp
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
#include "codegen/npu/litenpu/codegen_litenpu.h"

using namespace npu::tile_fwk;

class LiteNPUCodeGenFull : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

<<<<<<< HEAD
    void SetUp() override{
        == == == = void SetUp() override{
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
                     config::Reset();
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    config::SetBuildStatic(true);
}

    void TearDown() override
{}
}
;

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenFull, test_full_001)
{
    PROGRAM("FULL_001")
    {
        == == == = TEST_F(LiteNPUCodeGenFull, test_full_001)
        {
            PROGRAM("FULL_001")
            {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
                Element input(DataType::DT_FP16, 1.2);
                DataType dataType = DataType::DT_FP16;
                std::vector<int64_t> dstShape = {16, 16};

                auto output = Tensor(DataType::DT_FP16, {16, 16}, "output");
<<<<<<< HEAD
                FUNCTION("FULL_001")
                {
                    == == == = FUNCTION("FULL_001")
                    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
                        TileShape::Current().SetVecTile({16, 16});
                        output = Full(input, dataType, dstShape);
                    }
                }

                auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_001");
                npu::tile_fwk::CodeGenCtx ctx;
                npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
                codeGen.GenCode(*function, {});
            }
