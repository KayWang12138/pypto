/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_add_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct MaxSOpFuncArgs : public OpFuncArgs {
    MaxSOpFuncArgs(const Element &value, const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape)
        : value_(value), viewShape_(viewShape), tileShape_(tileShape) {}

    Element value_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct MaxSOpMetaData {
    explicit MaxSOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void MaxSOperationExeFuncDoubleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        auto args = static_cast<const MaxSOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = MaxS(tileTensor0, args->value_);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void MaxSOperationExeFuncTripleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        auto args = static_cast<const MaxSOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = MaxS(tileTensor0, args->value_);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void MaxSOperationExeFuncQuadrupleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const MaxSOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);
        int qloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(0, qloop, 1)) {
                        auto tileTensor0 =
                            View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                    std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                    std::min(fourthDim - qIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                    qIdx * fourthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = MaxS(tileTensor0, args->value_);
                        Assemble(res,
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                qIdx * fourthViewShape},
                            outputs[0]);
                    }
                }
            }
        }
    }
}

class MaxSOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<MaxSOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestMaxS, MaxSOperationTest,
    ::testing::ValuesIn(GetOpMetaData<MaxSOpMetaData>(
        {MaxSOperationExeFuncDoubleCut, MaxSOperationExeFuncTripleCut, MaxSOperationExeFuncQuadrupleCut}, "MaxS")));

Element GetElementByType(DataType dataType, nlohmann::json test_data, string name) {
    if (dataType == DT_FP32 || dataType == DT_FP16 || dataType == DT_BF16) {
        Element element(dataType, GetValueByName<float>(test_data, name));
        return element;
    } else if (dataType == DT_INT8) {
        Element element(dataType, GetValueByName<int8_t>(test_data, name));
        return element;
    } else if (dataType == DT_INT16) {
        Element element(dataType, GetValueByName<int16_t>(test_data, name));
        return element;
    } else if (dataType == DT_INT32) {
        Element element(dataType, GetValueByName<int32_t>(test_data, name));
        return element;
    } else if (dataType == DT_INT64) {
        Element element(dataType, GetValueByName<int64_t>(test_data, name));
        return element;
    } else {
        std::string errorMessage = "UnSupport Type in MaxS ST Test" + DataType2String(dataType);
        throw std::invalid_argument(errorMessage.c_str());
    }
}


TEST_P(MaxSOperationTest, TestMaxS) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    auto dtype = GetDataType(GetValueByName<std::string>(test_data, "scalar_type"));
    Element value = GetElementByType(dtype, test_data, "scalar");
    auto args = MaxSOpFuncArgs(value, GetViewShape(test_data), GetTileShape(test_data));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0].GetStorage()->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}
} // namespace