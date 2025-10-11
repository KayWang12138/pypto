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
 * \file test_log_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
using npu::tile_fwk::LogBaseType;
namespace {
struct LogOpFuncArgs : public OpFuncArgs {
    LogOpFuncArgs(const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape, const LogBaseType base)
        : viewShape_(viewShape), tileShape_(tileShape), base_(base) {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    LogBaseType base_;
};

struct LogOpMetaData {
    explicit LogOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void LogOperationExeFunc1Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        const struct LogOpFuncArgs *args = static_cast<const LogOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int bloop = CeilDiv(firstDim, firstViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            auto tileTensor = View(inputs[0], {firstViewShape},
                {std::min(firstDim - bIdx * firstViewShape, firstViewShape)},
                {bIdx * firstViewShape});
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Log(tileTensor, args->base_);
            Assemble(res, {bIdx * firstViewShape}, outputs[0]);
        }
    }
}

static void LogOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const struct LogOpFuncArgs *args = static_cast<const LogOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Log(tileTensor, args->base_);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void LogOperationExeFunc3Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        const struct LogOpFuncArgs *args = static_cast<const LogOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensor = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Log(tileTensor, args->base_);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void LogOperationExeFunc4Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        SymbolicScalar fourthDim = inputs[0]->shape[3];
        auto args = static_cast<const LogOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int mloop = CeilDiv(thirdDim, thirdViewShape);
        const int nloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1)) {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                        Tensor tileTensor0 =
                            View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                    std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                    std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                    nIdx * fourthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Log(tileTensor0, args->base_);
                        Assemble(res,
                            {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                nIdx * fourthViewShape},
                            outputs[0]);
                    }
                }
            }
        }
    }
}

class LogOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<LogOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestLog, LogOperationTest,
    ::testing::ValuesIn(GetOpMetaData<LogOpMetaData>(
        {LogOperationExeFunc2Dims, LogOperationExeFunc3Dims, LogOperationExeFunc4Dims, LogOperationExeFunc1Dims}, "Log")));

TEST_P(LogOperationTest, TestLog) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    std::string baseStr = GetValueByName<std::string>(test_data, "base");
    LogBaseType base;
    if (baseStr == "e") {
        base = LogBaseType::LOG_e;
    } else if (baseStr == "2") {
        base = LogBaseType::LOG_2;
    } else if (baseStr == "10") {
        base = LogBaseType::LOG_10;
    } else {
       assert(false && "unsupported base");
    }
    auto args = LogOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), base);
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0]->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0]->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}
} // namespace