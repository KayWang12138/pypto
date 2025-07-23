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

class DynamicMatmulTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template<typename InputT, typename OutputT, bool IsBtrans = false, bool IsBNZ = false>
void TestDynMatmul(int m, int k, int n, string dataPath) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    int ka = k;
    int kb = k;
    int nb = n;
    if constexpr (IsBtrans) {
        std::swap(kb, nb);
    }
    std::vector<int> shape_a = {m, ka};
    std::vector<int> shape_b = {kb, nb};
    std::vector<int> shape_c = {m, n};

    auto InputAstDtype = GetAstDtype<InputT>();
    auto OutputAstDtype = GetAstDtype<OutputT>();

    Tensor tensor_a(InputAstDtype, shape_a, "tensor_a");
    auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor tensor_b(InputAstDtype, shape_b, "tensor_b", NodeType::LOCAL, bfmt);
    Tensor tensor_c(OutputAstDtype, shape_c, "tensor_c");
    FUNCTION("test_dyn_mm", FunctionType::DYNAMIC, {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
            Tensor dyn_a = DViewPad(tensor_a, {m, ka}, {m, ka}, {batchId * m, 0});
            Tensor dyn_b = DViewPad(tensor_b, {kb, nb}, {kb, nb}, {0, 0});
            tensor_c = Matrix::Matmul<false, IsBtrans>(OutputAstDtype, dyn_a, dyn_b);
        }
    }

    std::vector<InputT> aData(m * k, 0);
    std::vector<InputT> bData(k * n, 0);
    std::vector<OutputT> golden(m * n, 0);

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

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (OutputT *)outs->data(), 0.001f));
}

TEST_F(DynamicMatmulTest, mm_A_B_ND_bf16) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    TestDynMatmul<npu::tile_fwk::bfloat16, float, false, false> (m, k, n, GetGoldenDir());
}

TEST_F(DynamicMatmulTest, mm_A_B_NZ_bf16) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int m = 16;
    int k = 32;
    int n = 512;
    TestDynMatmul<npu::tile_fwk::bfloat16, float, false, true> (m, k, n, GetGoldenDir());
}

TEST_F(DynamicMatmulTest, mm_A_Bt_ND_fp16) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int m = 128;
    int k = 257;
    int n = 511;
    TestDynMatmul<npu::tile_fwk::float16, float, true, false> (m, k, n, GetGoldenDir());
}

TEST_F(DynamicMatmulTest, mm_A_Bt_NZ_fp16) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int m = 1;
    int k = 512;
    int n = 256;
    TestDynMatmul<npu::tile_fwk::float16, float, true, true> (m, k, n, GetGoldenDir());
}

TEST_F(DynamicMatmulTest, mm_A_B_NZ_int8) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int m = 16;
    int k = 32;
    int n = 512;
    TestDynMatmul<int8_t, int32_t, false, true> (m, k, n, GetGoldenDir());
}

TEST_F(DynamicMatmulTest, mm_A_Bt_NZ_int8) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int m = 1;
    int k = 512;
    int n = 256;
    TestDynMatmul<int8_t, int32_t, true, true> (m, k, n, GetGoldenDir());
}

TEST_F(DynamicMatmulTest, mm_A_B_ND_bf16_tile1) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {256, 256}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    TestDynMatmul<npu::tile_fwk::bfloat16, float, false, false> (m, k, n, GetGoldenDir());
}

TEST_F(DynamicMatmulTest, mm_A_Bt_ND_fp16_tile2) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {512, 512}, {32, 32});
    int m = 16;
    int k = 512;
    int n = 512;
    TestDynMatmul<npu::tile_fwk::float16, float, true, false> (m, k, n, GetGoldenDir());
}

TEST_F(DynamicMatmulTest, mm_A_B_NZ_int8_tile3) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {512, 512});
    int m = 16;
    int k = 32;
    int n = 512;
    TestDynMatmul<int8_t, int32_t, false, true> (m, k, n, GetGoldenDir());
}

TEST_F(DynamicMatmulTest, mm_A_Bt_NZ_int8_tile4) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {64, 64}, {32, 32});
    int m = 1;
    int k = 512;
    int n = 256;
    TestDynMatmul<int8_t, int32_t, true, true> (m, k, n, GetGoldenDir());
}
}// namespace