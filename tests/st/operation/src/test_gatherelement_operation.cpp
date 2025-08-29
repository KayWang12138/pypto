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
 * \file test_GatherElement_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct GatherElementOpFuncArgs : public OpFuncArgs {
    GatherElementOpFuncArgs(const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape, int axis)
        : viewShape_(viewShape), tileShape_(tileShape), axis_(axis) {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    int axis_;
};

struct GatherElementOpMetaData {
    explicit GatherElementOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void GatherElementOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar src_firstDim = inputs[0]->shape[0];
        SymbolicScalar src_secondDim = inputs[0]->shape[1];
        SymbolicScalar idx_firstDim = inputs[1]->shape[0];
        SymbolicScalar idx_secondDim = inputs[1]->shape[1];
        auto args = static_cast<const GatherElementOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        /* gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分。切分以index为准
         * src axis不能切分，需要保证axis轴的viewshape=src的shape */
        const int bloop = CeilDiv(idx_firstDim, firstViewShape);
        const int sloop = CeilDiv(idx_secondDim, secondViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(src_secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                auto tileTensor1 = DViewPad(inputs[1], {firstViewShape, secondViewShape},
                    {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});

                Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
                auto res = GatherElement(tileTensor0, tileTensor1, args->axis_);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

class GatherElementOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<GatherElementOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestGatherElement, GatherElementOperationTest,
    ::testing::ValuesIn(GetOpMetaData<GatherElementOpMetaData>(
        {GatherElementOperationExeFunc2Dims}, "GatherElement")));

TEST_P(GatherElementOperationTest, TestGatherElement) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    auto axis = static_cast<CastMode>(GetValueByName<int>(test_data, "axis"));
    auto args = GatherElementOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), axis);
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0]->Symbol() + ".bin",
        GetGoldenDir() + "/" + testCase.inputTensors[1]->Symbol() + ".bin" };
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0]->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}
} // namespace
