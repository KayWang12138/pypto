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
 * \file test_ut_matmul.cpp
 * \brief Unit test for pass manager.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

namespace {

class DynamicMatmulUTest : public testing::Test {};

template<typename InputT, typename OutputT, bool IsBtrans = false, bool IsBNZ = false>
void TestDynMatmul(int m, int k, int n) {
    config::SetPlatformConfig("ENABLE_COST_MODEL", true);
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    int nb = n;
    int kb = k;
    int ka = k;
    if constexpr (IsBtrans) {
        std::swap(nb, kb);
    }
    std::vector<int> shape_c = {m, n};
    std::vector<int> shape_b = {kb, nb};
    std::vector<int> shape_a = {m, ka};

    auto InputUTDtype = (std::is_same<InputT, npu::tile_fwk::bfloat16>::value) ? DT_BF16 : DT_INT8;
    auto OutputUTDtype = (std::is_same<OutputT, float>::value) ? DT_FP32 : DT_INT32;

    Tensor tensor_c(OutputUTDtype, shape_c, "tensor_c");
    Tensor tensor_a(InputUTDtype, shape_a, "tensor_a");
    auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor tensor_b(InputUTDtype, shape_b, "tensor_b", NodeType::LOCAL, bfmt);
    FUNCTION("test_dyn_mm", FunctionType::DYNAMIC, {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
            Tensor dyn_a = DViewPad(tensor_a, {m, ka}, {m, ka}, {batchId * m, 0});
            Tensor dyn_b = DViewPad(tensor_b, {kb, nb}, {kb, nb}, {0, 0});
            tensor_c = Matrix::Matmul<false, IsBtrans>(OutputUTDtype, dyn_a, dyn_b);
        }
    }
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_bf16) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    TestDynMatmul<npu::tile_fwk::bfloat16, float, false, false> (m, k, n);
}

TEST_F(DynamicMatmulUTest, mm_A_B_NZ_bf16) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int m = 16;
    int k = 32;
    int n = 512;
    TestDynMatmul<npu::tile_fwk::bfloat16, float, false, true> (m, k, n);
}

TEST_F(DynamicMatmulUTest, mm_A_B_NZ_int8) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int m = 16;
    int k = 32;
    int n = 512;
    TestDynMatmul<int8_t, int32_t, false, true> (m, k, n);
}

TEST_F(DynamicMatmulUTest, mm_A_Bt_NZ_int8) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    int m = 1;
    int k = 512;
    int n = 256;
    TestDynMatmul<int8_t, int32_t, true, true> (m, k, n);
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_bf16_tile1) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {256, 256}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    TestDynMatmul<npu::tile_fwk::bfloat16, float, false, false> (m, k, n);
}

TEST_F(DynamicMatmulUTest, mm_A_B_NZ_int8_tile3) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {512, 512});
    int m = 16;
    int k = 32;
    int n = 512;
    TestDynMatmul<int8_t, int32_t, false, true> (m, k, n);
}

TEST_F(DynamicMatmulUTest, mm_A_Bt_NZ_int8_tile4) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {64, 64}, {32, 32});
    int m = 1;
    int k = 512;
    int n = 256;
    TestDynMatmul<int8_t, int32_t, true, true> (m, k, n);
}
}// namespace