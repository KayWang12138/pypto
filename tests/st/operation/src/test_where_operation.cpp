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
 * \file test_add_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct WhereOpFuncArgs : public OpFuncArgs {
    WhereOpFuncArgs(const int flag, const Element &x_scalar, const Element &y_scalar, const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape)
        : flag_(flag), x_scalar_(x_scalar), y_scalar_(y_scalar), viewShape_(viewShape), tileShape_(tileShape) {}
    int flag_;
    Element x_scalar_;
    Element y_scalar_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct WhereOpMetaData {
    explicit WhereOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void WhereOperationExeFuncDoubleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
        config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);

        FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]}) {
            SymbolicScalar firstDim = max(inputs[0]->shape[0], max(inputs[1]->shape[0], inputs[2]->shape[0]));
            SymbolicScalar secondDim = max(inputs[0]->shape[1], max(inputs[1]->shape[1], inputs[2]->shape[1]));
            auto args = static_cast<const WhereOpFuncArgs *>(opArgs);
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];
            int bloop = CeilDiv(firstDim, firstViewShape);
            int sloop = CeilDiv(secondDim, secondViewShape);
            const int broadcastFlag = 1;
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                    Tensor tileTensor0;
                    Tensor tileTensor1;
                    Tensor tileTensor2;
                    if(inputs[0]->shape[1] != broadcastFlag && inputs[1]->shape[1] != broadcastFlag && inputs[2]->shape[1] == broadcastFlag) {
                        tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                        tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                        tileTensor2 = View(inputs[2], {firstViewShape, 1},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1}, {bIdx * firstViewShape, 0});
                    }
                    else if (inputs[0]->shape[0] != broadcastFlag && inputs[1]->shape[0] != broadcastFlag && inputs[2]->shape[0] == broadcastFlag) {
                        tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                        tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                        tileTensor2 = View(inputs[2], {1, secondViewShape},
                            {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {0, sIdx * secondViewShape});
                    }
                    else if (inputs[0]->shape[1] != broadcastFlag && inputs[1]->shape[1] == broadcastFlag && inputs[2]->shape[1] != broadcastFlag) {
                        tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                        tileTensor1 = View(inputs[1], {firstViewShape, 1},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1}, {bIdx * firstViewShape, 0});
                        tileTensor2 = View(inputs[2], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                    }
                    else if (inputs[0]->shape[0] != broadcastFlag && inputs[1]->shape[0] == broadcastFlag && inputs[2]->shape[0] != broadcastFlag) {
                        tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                        tileTensor1 = View(inputs[1], {1, secondViewShape},
                            {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {0, sIdx * secondViewShape});
                        tileTensor2 = View(inputs[2], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                    }
                    else if (inputs[0]->shape[1] == broadcastFlag && inputs[1]->shape[1] != broadcastFlag && inputs[2]->shape[1] != broadcastFlag) {
                        tileTensor0 = View(inputs[0], {firstViewShape, 1},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1}, {bIdx * firstViewShape, 0});
                        tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                        tileTensor2 = View(inputs[2], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                    }
                    else if (inputs[0]->shape[0] == broadcastFlag && inputs[1]->shape[0] != broadcastFlag && inputs[2]->shape[0] != broadcastFlag) {
                        tileTensor0 = View(inputs[0], {1, secondViewShape},
                            {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {0, sIdx * secondViewShape});
                        tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                        tileTensor2 = View(inputs[2], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                    }
                    else {
                        tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                        tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                        tileTensor2 = View(inputs[2], {firstViewShape, secondViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape});
                    }
                    int TT = 0;
                    int TS = 1;
                    int ST = 2;
                    int SS = 3;
                    TileShape::Current().SetVecTile(args->tileShape_);
                    if (args->flag_ == TT) {
                        auto res = Where(tileTensor0, tileTensor1, tileTensor2);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                    } else if (args->flag_ == TS) {
                        auto res = Where(tileTensor0, tileTensor1, args->y_scalar_);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                    } else if (args->flag_ == ST) {
                        auto res = Where(tileTensor0, args->x_scalar_, tileTensor2);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                    } else if (args->flag_ == SS) {
                        auto res = Where(tileTensor0, args->x_scalar_, args->y_scalar_);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                    }
                }
            }
        }
}


static void WhereOperationExeFuncTripleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);

    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        auto *args = static_cast<const WhereOpFuncArgs *>(opArgs);
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
                    auto tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    auto tileTensor2 = View(inputs[2], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    int TT = 0;
                    int TS = 1;
                    int ST = 2;
                    int SS = 3;
                    if (args->flag_ == TT) {
                        auto res = Where(tileTensor0, tileTensor1, tileTensor2);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                    } else if (args->flag_ == TS) {
                        auto res = Where(tileTensor0, tileTensor1, args->y_scalar_);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                    } else if (args->flag_ == ST) {
                        auto res = Where(tileTensor0, args->x_scalar_, tileTensor2);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                    } else if (args->flag_ == SS) {
                        auto res = Where(tileTensor0, args->x_scalar_, args->y_scalar_);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

static void WhereOperationExeFuncQuadrupleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);

    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        SymbolicScalar fourthDim = inputs[0]->shape[3];
        auto args = static_cast<const WhereOpFuncArgs *>(opArgs);
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
                        auto tileTensor1 =
                            View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                    std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                    std::min(fourthDim - qIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                    qIdx * fourthViewShape});
                        auto tileTensor2 =
                            View(inputs[2], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                    std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                    std::min(fourthDim - qIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                    qIdx * fourthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        int TT = 0;
                        int TS = 1;
                        int ST = 2;
                        int SS = 3;
                        if (args->flag_ == TT) {
                            auto res = Where(tileTensor0, tileTensor1, tileTensor2);
                            Assemble(res,
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                qIdx * fourthViewShape}, outputs[0]);
                        } else if (args->flag_ == TS) {
                            auto res = Where(tileTensor0, tileTensor1, args->y_scalar_);
                            Assemble(res,
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                qIdx * fourthViewShape}, outputs[0]);
                        } else if (args->flag_ == ST) {
                            auto res = Where(tileTensor0, args->x_scalar_, tileTensor2);
                            Assemble(res,
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                qIdx * fourthViewShape}, outputs[0]);
                        } else if (args->flag_ == SS) {
                            auto res = Where(tileTensor0, args->x_scalar_, args->y_scalar_);
                            Assemble(res,
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                qIdx * fourthViewShape}, outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

class WhereOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<WhereOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestWhere, WhereOperationTest,
    ::testing::ValuesIn(GetOpMetaData<WhereOpMetaData>(
        {WhereOperationExeFuncDoubleCut, WhereOperationExeFuncTripleCut, WhereOperationExeFuncQuadrupleCut}, "Where")));

TEST_P(WhereOperationTest, TestWhere) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    auto dtypeFlag = GetDataType(GetValueByName<std::string>(test_data, "flag_dtype"));
    Element flagElement(dtypeFlag, GetValueByName<int64_t>(test_data, "flag"));
    int flag = flagElement.GetSignedData();
    auto dtype = GetDataType(GetValueByName<std::string>(test_data, "scalar_dtype"));
    Element x_scalar(dtype, GetValueByName<float>(test_data, "x_scalar"));
    Element y_scalar(dtype, GetValueByName<float>(test_data, "y_scalar"));
    auto args = WhereOpFuncArgs(flag, x_scalar, y_scalar, GetViewShape(test_data), GetTileShape(test_data));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0]->Symbol() + ".bin",
        GetGoldenDir() + "/" + testCase.inputTensors[1]->Symbol() + ".bin",
        GetGoldenDir() + "/" + testCase.inputTensors[2]->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0]->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}
} // namespace