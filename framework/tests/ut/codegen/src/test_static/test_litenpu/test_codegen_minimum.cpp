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
 * \file test_codegen_minimum.cpp
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

class LiteNPUCodeGenMinimum : public testing::Test {
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
TEST_F(LiteNPUCodeGenMinimum, test_minimum_001)
{
    PROGRAM("MINIMUM_001")
    {
        == == == = TEST_F(LiteNPUCodeGenMinimum, test_minimum_001)
        {
            PROGRAM("MINIMUM_001")
            {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
                TileShape::Current().SetVecTile({4, 4});
                Tensor input(DT_FP32, {8, 8}, "input");
                Tensor other(DT_FP32, {8, 8}, "other");
                std::vector<int64_t> dstShape = {8, 8};
                Tensor output;
<<<<<<< HEAD
                FUNCTION("MINIMUM_001") { output = Minimum(input, other); }
                == == == = FUNCTION("MINIMUM_001") { output = Minimum(input, other); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
            }

            auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_001");
            npu::tile_fwk::CodeGenCtx ctx;
            npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
            codeGen.GenCode(*function, {});
        }

<<<<<<< HEAD
        TEST_F(LiteNPUCodeGenMinimum, test_minimum_002)
        {
            PROGRAM("MINIMUM_001")
            {
                == == == = TEST_F(LiteNPUCodeGenMinimum, test_minimum_002)
                {
                    PROGRAM("MINIMUM_001")
                    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
                        TileShape::Current().SetVecTile({4, 4});
                        Tensor input(DT_FP32, {8, 8}, "input");
                        Tensor other(DT_FP32, {1, 8}, "other");
                        std::vector<int64_t> dstShape = {8, 8};
                        Tensor output;
<<<<<<< HEAD
                        FUNCTION("MINIMUM_001") { output = Minimum(input, other); }
                        == == == = FUNCTION("MINIMUM_001") { output = Minimum(input, other); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
                    }

                    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_001");
                    npu::tile_fwk::CodeGenCtx ctx;
                    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
                    codeGen.GenCode(*function, {});
                }

<<<<<<< HEAD
                TEST_F(LiteNPUCodeGenMinimum, test_minimum_003)
                {
                    PROGRAM("MINIMUM_001")
                    {
                        == == == = TEST_F(LiteNPUCodeGenMinimum, test_minimum_003)
                        {
                            PROGRAM("MINIMUM_001")
                            {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
                                TileShape::Current().SetVecTile({4, 4});
                                Tensor input(DT_FP32, {8, 8}, "input");
                                Element other(DT_FP32, 8.0);
                                std::vector<int64_t> dstShape = {8, 8};
                                Tensor output;
<<<<<<< HEAD
                                FUNCTION("MINIMUM_001") { output = Minimum(input, other); }
                                == == == = FUNCTION("MINIMUM_001") { output = Minimum(input, other); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
                            }

                            auto function =
                                Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_001");
                            npu::tile_fwk::CodeGenCtx ctx;
                            npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
                            codeGen.GenCode(*function, {});
<<<<<<< HEAD
                        }
                        == == == =
                    }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
