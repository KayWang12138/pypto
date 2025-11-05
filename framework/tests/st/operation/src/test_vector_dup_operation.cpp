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
 * \file test_vector_dup_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct VectorDuplicateOpFuncArgs : public OpFuncArgs {
    VectorDuplicateOpFuncArgs(const Element &value, const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape)
        : value_(value), viewShape_(viewShape), tileShape_(tileShape) {}

    Element value_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct VectorDuplicateOpMetaData {
    explicit VectorDuplicateOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void VectorDuplicateOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        auto args = static_cast<const VectorDuplicateOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        TileShape::Current().SetVecTile(args->tileShape_);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor = VectorDuplicate(args->value_, DataType::DT_FP32, {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)});
                Assemble(tileTensor, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void VectorDuplicateOperationExeFunc3Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        auto args = static_cast<const VectorDuplicateOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);
        TileShape::Current().SetVecTile(args->tileShape_);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensor = VectorDuplicate(args->value_, DataType::DT_FP32,
                        {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)});
                    Assemble(
                        tileTensor, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void VectorDuplicateOperationExeFunc4Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const VectorDuplicateOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int mloop = CeilDiv(thirdDim, thirdViewShape);
        const int nloop = CeilDiv(fourthDim, fourthViewShape);

        TileShape::Current().SetVecTile(args->tileShape_);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1)) {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                        Tensor tileTensor0 = VectorDuplicate(args->value_, DataType::DT_FP32,
                            {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)});
                        Assemble(tileTensor0,
                            {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                nIdx * fourthViewShape},
                            outputs[0]);
                    }
                }
            }
        }
    }
}

class VectorDuplicateOperationTest
    : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<VectorDuplicateOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestVectorDuplicate, VectorDuplicateOperationTest,
    ::testing::ValuesIn(GetOpMetaData<VectorDuplicateOpMetaData>(
        {VectorDuplicateOperationExeFunc2Dims, VectorDuplicateOperationExeFunc3Dims,
            VectorDuplicateOperationExeFunc4Dims},
        "VectorDuplicate")));

TEST_P(VectorDuplicateOperationTest, TestVectorDuplicate) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    auto dtype = GetDataType(GetValueByName<std::string>(test_data, "scalar_type"));
    Element value(dtype, GetValueByName<float>(test_data, "scalar"));
    auto args = VectorDuplicateOpFuncArgs(value, GetViewShape(test_data), GetTileShape(test_data));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0].GetStorage()->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}
} // namespace
