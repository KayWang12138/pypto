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
 * \file test_interp_type_convert.cpp
 * \brief Unit tests for interpreter internal type conversion, especially FP8 (E4M3, E5M2, E8M0).
 */

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk {

template <typename T>
static LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int64_t> &shape, const T &val) {
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor(data, val));
}

template <typename T>
static LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int64_t> &shape,
                                           const std::vector<T> &vals) {
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateTensor(data, vals));
}

#define ASSERT_ALLCLOSE(self, other) \
    ASSERT(calc::AllClose(self, other)) << "lhs:\n" << self->ToString() << "\nrhs:\n" << other->ToString() << "\n"

#define ASSERT_ALLCLOSE_ATOL(lhs, rhs, atol) \
    ASSERT(calc::AllClose(lhs, rhs, atol, 1e-5)) << "lhs:\n" << lhs->ToString() << "\nrhs:\n" << rhs->ToString() << "\n"

class InterpTypeConvertTest : public testing::Test {
public:
    void SetUp() override {
        if (!calc::IsVerifyEnabled()) {
            GTEST_SKIP() << "Torch verifier not enabled, skip type conversion tests";
        }
        Program::GetInstance().Reset();
        config::Reset();
    }
};

// 0x55 = 0101 0101: same 8 bits, different interpretations per format:
//   E4M3 [S][EEEE][MMM]: S=0, E=1010=10, M=101=5 -> 2^(10-7)*(1+5/8) = 13.0
//   E5M2 [S][EEEEE][MM]: S=0, E=10101=21, M=01=1 -> 2^(21-15)*(1+1/4) = 80.0
//   E8M0 [S][EEEEEEE]:   S=0, E=1010101=85 -> 2^(85-63) = 4194304
TEST_F(InterpTypeConvertTest, Fp8SameBitsDifferentFormats) {
    constexpr uint8_t kBits = 0x55;

    auto e4m3_src = makeTensorData(DT_FP8E4M3, {4}, static_cast<uint8_t>(kBits));
    auto e5m2_src = makeTensorData(DT_FP8E5M2, {4}, static_cast<uint8_t>(kBits));
    auto e8m0_src = makeTensorData(DT_FP8E8M0, {4}, static_cast<uint8_t>(kBits));

    auto e4m3_out = makeTensorData(DT_FP32, {4}, 0.0f);
    auto e5m2_out = makeTensorData(DT_FP32, {4}, 0.0f);
    auto e8m0_out = makeTensorData(DT_FP32, {4}, 0.0f);

    calc::Cast(e4m3_out, e4m3_src);
    calc::Cast(e5m2_out, e5m2_src);
    calc::Cast(e8m0_out, e8m0_src);

    auto golden_e4m3 = makeTensorData(DT_FP32, {4}, 13.0f);
    auto golden_e5m2 = makeTensorData(DT_FP32, {4}, 80.0f);
    auto golden_e8m0 = makeTensorData(DT_FP32, {4}, 4194304.0f);

    ASSERT_ALLCLOSE_ATOL(e4m3_out, golden_e4m3, 1e-6f);
    ASSERT_ALLCLOSE_ATOL(e5m2_out, golden_e5m2, 1e-6f);
    ASSERT_ALLCLOSE_ATOL(e8m0_out, golden_e8m0, 1e-6f);
}

// 0x01 = 0000 0001: same 8 bits, interpreted as subnormal in E4M3/E5M2
//   E4M3 [S][EEEE][MMM]: S=0, E=0000=0, M=001=1 -> 2^(-6) * (1/8) = 1/512 ≈ 0.001953125
//   E5M2 [S][EEEEE][MM]: S=0, E=00000=0, M=01=1 -> 2^(-14) * (1/4) = 1/65536 ≈ 0.000015258789
//   E8M0 [S][EEEEEEEE]:  S=0, E=00000001=1 -> 2^(1-63) = 2^(-62) ≈ 2.1684e-19 (normal, not subnormal)
TEST_F(InterpTypeConvertTest, Fp8SubnormalSameBitsDifferentFormats) {
    constexpr uint8_t kBits = 0x01;

    auto e4m3_src = makeTensorData(DT_FP8E4M3, {4}, static_cast<uint8_t>(kBits));
    auto e5m2_src = makeTensorData(DT_FP8E5M2, {4}, static_cast<uint8_t>(kBits));
    auto e8m0_src = makeTensorData(DT_FP8E8M0, {4}, static_cast<uint8_t>(kBits));

    auto e4m3_out = makeTensorData(DT_FP32, {4}, 0.0f);
    auto e5m2_out = makeTensorData(DT_FP32, {4}, 0.0f);
    auto e8m0_out = makeTensorData(DT_FP32, {4}, 0.0f);

    calc::Cast(e4m3_out, e4m3_src);
    calc::Cast(e5m2_out, e5m2_src);
    calc::Cast(e8m0_out, e8m0_src);

    const float e4m3_golden = 1.0f / 512.0f;           // ≈ 0.001953125f
    const float e5m2_golden = 1.0f / 65536.0f;         // ≈ 0.000015258789f
    const float e8m0_golden = std::exp2(-62.0f);       // ≈ 2.1684043e-19f

    auto golden_e4m3 = makeTensorData(DT_FP32, {4}, e4m3_golden);
    auto golden_e5m2 = makeTensorData(DT_FP32, {4}, e5m2_golden);
    auto golden_e8m0 = makeTensorData(DT_FP32, {4}, e8m0_golden);

    ASSERT_ALLCLOSE_ATOL(e4m3_out, golden_e4m3, 1e-8f);
    ASSERT_ALLCLOSE_ATOL(e5m2_out, golden_e5m2, 1e-10f); // 更小值，需更紧容差或用 rel tol
    ASSERT_ALLCLOSE_ATOL(e8m0_out, golden_e8m0, 1e-22f);
}
// FP8 E4M3 special values: +0(0x00), -0(0x80), +Inf(0x7E), -Inf(0xFE), NaN(0x7F)
TEST_F(InterpTypeConvertTest, Fp8SpecialValues) {
    struct {
        uint8_t enc;
        float expected;
        bool is_nan;
    } cases[] = {
        {0x00, 0.0f, false},
        {0x80, -0.0f, false},
        {0x7E, std::numeric_limits<float>::infinity(), false},
        {0xFE, -std::numeric_limits<float>::infinity(), false},
        {0x7F, 0.0f, true},
    };

    for (const auto &c : cases) {
        auto src = makeTensorData(DT_FP8E4M3, {4}, static_cast<uint8_t>(c.enc));
        auto out = makeTensorData(DT_FP32, {4}, 0.0f);
        calc::Cast(out, src);

        if (c.is_nan) {
            for (int64_t i = 0; i < 4; ++i) {
                ASSERT(std::isnan(out->Get<float>(i))) << "expected NaN at index " << i;
            }
        } else {
            auto golden = makeTensorData(DT_FP32, {4}, c.expected);
            ASSERT_ALLCLOSE(out, golden);
        }
    }
}

}  // namespace npu::tile_fwk
