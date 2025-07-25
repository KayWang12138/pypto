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
 * \file test_dynamic_mm.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "operation/tilefwk_op.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dynamic.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {

class DynamicBatchMatmulTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template<typename InputT, typename OutputT, bool IsBtrans = false, bool IsBNZ = false>
void TestDynBatchMatmul(int b, int m, int k, int n, string dataPath) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    int ka = k;
    int kb = k;
    int nb = n;
    if constexpr (IsBtrans) {
        std::swap(kb, nb);
    }
    std::vector<int> shape_a = {b, m, ka};
    std::vector<int> shape_b = {b, kb, nb};
    std::vector<int> shape_c = {b, m, n};

    auto InputAstDtype = GetAstDtype<InputT>();
    auto OutputAstDtype = GetAstDtype<OutputT>();

    Tensor tensor_a(InputAstDtype, shape_a, "tensor_a");
    auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor tensor_b(InputAstDtype, shape_b, "tensor_b", NodeType::LOCAL, bfmt);
    Tensor tensor_c(OutputAstDtype, shape_c, "tensor_c");

    FUNCTION("test_dyn_bmm", FunctionType::DYNAMIC, {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1)) {
            Tensor dyn_a = DViewPad(tensor_a, {b, m, ka}, {b, m, ka}, {0, mIdx, 0});
            Tensor dyn_b = DViewPad(tensor_b, {b, kb, nb}, {b, kb, nb}, {0, 0, 0});
            if constexpr (IsBNZ) {
                Program::GetInstance().GetMatrixSize().SetMatrixSize({m, k, n});
            }
            tensor_c = Matrix::BatchMatmul<false, IsBtrans>(OutputAstDtype, dyn_a, dyn_b);
        }
    }

    std::vector<InputT> aData(b * m * k, 0);
    std::vector<InputT> bData(b * k * n, 0);
    std::vector<OutputT> golden(b * m * n, 0);

    readInput<InputT>(dataPath + "/mat_a.bin", aData);
    readInput<InputT>(dataPath + "/mat_b.bin", bData);
    readInput<OutputT>(dataPath + "/mat_c.bin", golden);

     ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<InputT>(tensor_a, aData),
        RawTensorData::CreateTensor<InputT>(tensor_b, bData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<OutputT>(tensor_c, 0.0f),
    });

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (OutputT *)outs->data(), 0.001f));
}

TEST_F(DynamicBatchMatmulTest, test_bmm_A_B_ND_bf16) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int b = 3;
    int m = 64;
    int k = 128;
    int n = 512;
    TestDynBatchMatmul<npu::tile_fwk::bfloat16, float, false, false> (b, m, k, n, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulTest, test_bmm_A_Bt_ND_fp16) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int b = 3;
    int m = 2;
    int k = 576;
    int n = 4096;
    TestDynBatchMatmul<npu::tile_fwk::float16, float, true, false> (b, m, k, n, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulTest, test_bmm_A_B_NZ_bf16) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int b = 2;
    int m = 16;
    int k = 512;
    int n = 128;
    TestDynBatchMatmul<npu::tile_fwk::bfloat16, float, false, true> (b, m, k, n, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulTest, test_bmm_A_Bt_NZ_fp16) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int b = 4;
    int m = 96;
    int k = 128;
    int n = 256;
    TestDynBatchMatmul<npu::tile_fwk::float16, float, true, true> (b, m, k, n, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulTest, test_bmm_A_B_ND_bf16_tile1) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {256, 256}, {128, 128});
    int b = 6;
    int m = 1;
    int k = 576;
    int n = 4096;
    TestDynBatchMatmul<npu::tile_fwk::bfloat16, float, false, false> (b, m, k, n, GetGoldenDir());
}
}// namespace