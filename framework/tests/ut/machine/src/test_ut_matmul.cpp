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

template <typename T>
DataType GetAstDtype() {
    DataType astDtype = DataType::DT_BOTTOM;
    if constexpr (std::is_same<T, npu::tile_fwk::float16>::value) {
        astDtype = DataType::DT_FP16;
    }
    if constexpr (std::is_same<T, float>::value) {
        astDtype = DataType::DT_FP32;
    }
    if constexpr (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        astDtype = DataType::DT_BF16;
    }
    if constexpr (std::is_same<T, int8_t>::value) {
        astDtype = DT_INT8;
    }
    if constexpr (std::is_same<T, int32_t>::value) {
        astDtype = DT_INT32;
    }
    EXPECT_NE(astDtype, DT_BOTTOM);
    return astDtype;
}

template <typename InputT, typename OutputT, bool IsBtrans = false, bool IsBNZ = false>
void TestDynMatmul(int m, int k, int n, Matrix::MatmulExtendParam param = {}) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    int nb = n;
    int kb = k;
    int ka = k;
    if constexpr (IsBtrans) {
        std::swap(nb, kb);
    }
    std::vector<int64_t> shape_c = {m, n};
    std::vector<int64_t> shape_b = {kb, nb};
    std::vector<int64_t> shape_a = {m, ka};

    auto InputUTDtype = GetAstDtype<InputT>();
    auto OutputUTDtype = GetAstDtype<OutputT>();

    Tensor tensor_c(OutputUTDtype, shape_c, "tensor_c");
    Tensor tensor_a(InputUTDtype, shape_a, "tensor_a");
    auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor tensor_b(InputUTDtype, shape_b, "tensor_b", bfmt);
    FUNCTION("test_dyn_mm", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
            Tensor dyn_a = View(tensor_a, {m, ka}, {m, ka}, {batchId * m, 0});
            Tensor dyn_b = View(tensor_b, {kb, nb}, {kb, nb}, {0, 0});
            tensor_c = Matrix::Matmul<false, IsBtrans>(OutputUTDtype, dyn_a, dyn_b, param);
        }
    }
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_bf16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    TestDynMatmul<npu::tile_fwk::bfloat16, float, false, false> (m, k, n);
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_pertensor) {
    int m = 128;
    int n = 512;
    int k = 256;
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Matrix::MatmulExtendParam param;
    param.scaleValue = 2.0f;
    TestDynMatmul<int8_t, npu::tile_fwk::float16, false, false> (m, k, n, param);
}
}// namespace