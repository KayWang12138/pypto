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
 * \file test_scatter_tensor_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace ScatterOperation {
extern const std::map<std::string, ScatterMode>& GetScatterModeMap();
}
namespace ScatterTensorOperation {
struct ScatterTensorOpFuncArgs : public OpFuncArgs {
    ScatterTensorOpFuncArgs(
        const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, int axis, ScatterMode reduce)
        : viewShape_(viewShape), tileShape_(tileShape), axis_(axis), reduce_(reduce)
    {
        this->inplaceInfo[0] = 0; // 表示第0个输出初始化为第0个输入的数据
    }

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    int axis_;
    ScatterMode reduce_;
};

struct ScatterTensorOpMetaData {
    explicit ScatterTensorOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void ScatterTensorOperationExeFunc2Dims(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        auto args = static_cast<const ScatterTensorOpFuncArgs*>(opArgs);
        TileShape::Current().SetVecTile(args->tileShape_);
        Scatter(outputs[0], inputs[1], inputs[2], args->axis_, args->reduce_);
    }
}

static void ScatterTensorOperationExeFunc3Dims(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        auto args = static_cast<const ScatterTensorOpFuncArgs*>(opArgs);
        TileShape::Current().SetVecTile(args->tileShape_);
        Scatter(outputs[0], inputs[1], inputs[2], args->axis_, args->reduce_);
    }
}

static void ScatterTensorOperationExeFunc4Dims(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        auto args = static_cast<const ScatterTensorOpFuncArgs*>(opArgs);
        TileShape::Current().SetVecTile(args->tileShape_);
        Scatter(outputs[0], inputs[1], inputs[2], args->axis_, args->reduce_);
    }
}

class ScatterTensorOperationTest
    : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ScatterTensorOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestScatterTensor, ScatterTensorOperationTest,
    ::testing::ValuesIn(
        GetOpMetaData<ScatterTensorOpMetaData>(
            {ScatterTensorOperationExeFunc2Dims, ScatterTensorOperationExeFunc3Dims,
             ScatterTensorOperationExeFunc4Dims},
            "ScatterTensor")));

TEST_P(ScatterTensorOperationTest, TestScatterTensor)
{
    auto test_data = GetParam().test_data_;
    auto axis = GetValueByName<int>(test_data, "axis");
    auto reduce =
        GetMapValByName(ScatterOperation::GetScatterModeMap(), GetValueByName<std::string>(test_data, "reduce"));
    auto args = ScatterTensorOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), axis, reduce);
    auto testCase = CreateTestCaseDesc<ScatterTensorOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace ScatterTensorOperation
