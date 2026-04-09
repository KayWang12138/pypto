/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_mxquant_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct QuantMXOpFuncArgs : public OpFuncArgs {
    explicit QuantMXOpFuncArgs(const std::vector<int64_t>& tileShape) : tileShape_(tileShape) {}

    std::vector<int64_t> tileShape_;
};

struct QuantMXOpMetaData {
    explicit QuantMXOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void QuantMXOperationExeFunc(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        auto args = static_cast<const QuantMXOpFuncArgs*>(opArgs);
        TileShape::Current().SetVecTile(args->tileShape_);
        auto res = QuantMX(inputs[0]);
        Reshape(std::get<0>(res), outputs[0]);
        Reshape(std::get<1>(res), outputs[1]);
    }
}

class QuantMXOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<QuantMXOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestQuantMX, QuantMXOperationTest,
    ::testing::ValuesIn(GetOpMetaData<QuantMXOpMetaData>({QuantMXOperationExeFunc}, "QuantMX")));

TEST_P(QuantMXOperationTest, TestQuantMX)
{
    auto args = QuantMXOpFuncArgs(GetTileShape(GetParam().test_data_));
    auto testCase = CreateTestCaseDesc<QuantMXOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
