/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_bitwise_xor_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct BitwiseXorOpFuncArgs : public OpFuncArgs {
    BitwiseXorOpFuncArgs(const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape) {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct BitwiseXorOpMetaData {
    explicit BitwiseXorOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

// 辅助函数：创建视图
static Tensor CreateViewForDim(const Tensor& input, const std::vector<int64_t>& baseViewShape,
                               const std::vector<SymbolicScalar>& baseFullSize,
                               const std::vector<SymbolicScalar>& baseFullOffset,
                               int broadcastDim) {
    std::vector<int64_t> viewShape = baseViewShape;
    std::vector<SymbolicScalar> fullSize = baseFullSize;
    std::vector<SymbolicScalar> fullOffset = baseFullOffset;
    
    if (broadcastDim >= 0 && input.GetShape()[broadcastDim] == 1) {
        viewShape[broadcastDim] = 1;
        fullSize[broadcastDim] = 1;
        fullOffset[broadcastDim] = 0;
    }
    return View(input, viewShape, fullSize, fullOffset);
}

static void BitwiseXorOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {

    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const BitwiseXorOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        // 检查不同维度并找出广播维度
        int diffCount = 0, broadcastDim = -1;
        for (int dim = 0; dim < 2; ++dim) {
            int s0 = inputs[0].GetShape()[dim], s1 = inputs[1].GetShape()[dim];
            if (s0 != s1) { diffCount++; if (s0 == 1 || s1 == 1) broadcastDim = dim; }
        }
        if (diffCount > 1) std::cout << "Multiple different dimensions detected" << std::endl;
        
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto fullSize0 = std::min(firstDim - bIdx * firstViewShape, firstViewShape);
                auto fullSize1 = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                auto fullOffset0 = bIdx * firstViewShape;
                auto fullOffset1 = sIdx * secondViewShape;

                std::vector<int64_t> baseView = {firstViewShape, secondViewShape};
                std::vector<SymbolicScalar> baseSize = {fullSize0, fullSize1};
                std::vector<SymbolicScalar> baseOffset = {fullOffset0, fullOffset1};
                
                Tensor tileTensor0, tileTensor1;
                if (diffCount == 1 && broadcastDim != -1) {
                    tileTensor0 = CreateViewForDim(inputs[0], baseView, baseSize, baseOffset, 
                                                  inputs[0].GetShape()[broadcastDim] == 1 ? broadcastDim : -1);
                    tileTensor1 = CreateViewForDim(inputs[1], baseView, baseSize, baseOffset, 
                                                  inputs[1].GetShape()[broadcastDim] == 1 ? broadcastDim : -1);
                } else {
                    tileTensor0 = View(inputs[0], baseView, baseSize, baseOffset);
                    tileTensor1 = View(inputs[1], baseView, baseSize, baseOffset);
                }

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = BitwiseXor(tileTensor0, tileTensor1);
                Assemble(res, {fullOffset0, fullOffset1}, outputs[0]);
            }
        }
    }
}

static void BitwiseXorOperationExeFunc3Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {

    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const BitwiseXorOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];

        int diffCount = 0, broadcastDim = -1;
        for (int dim = 0; dim < 3; ++dim) {
            int s0 = inputs[0].GetShape()[dim], s1 = inputs[1].GetShape()[dim];
            if (s0 != s1) { diffCount++; if (s0 == 1 || s1 == 1) broadcastDim = dim; }
        }
        if (diffCount > 1) std::cout << "Multiple different dimensions detected" << std::endl;
        
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        SymbolicScalar thirdDim = std::max(inputs[0].GetShape()[2], inputs[1].GetShape()[2]);

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto fullSize0 = std::min(firstDim - bIdx * firstViewShape, firstViewShape);
                    auto fullSize1 = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                    auto fullSize2 = std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape);
                    auto fullOffset0 = bIdx * firstViewShape;
                    auto fullOffset1 = sIdx * secondViewShape;
                    auto fullOffset2 = nIdx * thirdViewShape;

                    std::vector<int64_t> baseView = {firstViewShape, secondViewShape, thirdViewShape};
                    std::vector<SymbolicScalar> baseSize = {fullSize0, fullSize1, fullSize2};
                    std::vector<SymbolicScalar> baseOffset = {fullOffset0, fullOffset1, fullOffset2};
                    
                    Tensor tileTensor0, tileTensor1;
                    if (diffCount == 1 && broadcastDim != -1) {
                        tileTensor0 = CreateViewForDim(inputs[0], baseView, baseSize, baseOffset,
                                                      inputs[0].GetShape()[broadcastDim] == 1 ? broadcastDim : -1);
                        tileTensor1 = CreateViewForDim(inputs[1], baseView, baseSize, baseOffset,
                                                      inputs[1].GetShape()[broadcastDim] == 1 ? broadcastDim : -1);
                    } else {
                        tileTensor0 = View(inputs[0], baseView, baseSize, baseOffset);
                        tileTensor1 = View(inputs[1], baseView, baseSize, baseOffset);
                    }

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = BitwiseXor(tileTensor0, tileTensor1);
                    Assemble(res, {fullOffset0, fullOffset1, fullOffset2}, outputs[0]);
                }
            }
        }
    }
}

static void BitwiseXorOperationExeFunc4Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const BitwiseXorOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

        int diffCount = 0, broadcastDim = -1;
        for (int dim = 0; dim < 4; ++dim) {
            int s0 = inputs[0].GetShape()[dim], s1 = inputs[1].GetShape()[dim];
            if (s0 != s1) { diffCount++; if (s0 == 1 || s1 == 1) broadcastDim = dim; }
        }
        if (diffCount > 1) std::cout << "Multiple different dimensions detected" << std::endl;
        
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        SymbolicScalar thirdDim = std::max(inputs[0].GetShape()[2], inputs[1].GetShape()[2]);
        SymbolicScalar fourthDim = std::max(inputs[0].GetShape()[3], inputs[1].GetShape()[3]);

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int mloop = CeilDiv(thirdDim, thirdViewShape);
        const int nloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1)) {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                        auto fullSize0 = std::min(firstDim - bIdx * firstViewShape, firstViewShape);
                        auto fullSize1 = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                        auto fullSize2 = std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape);
                        auto fullSize3 = std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape);
                        auto fullOffset0 = bIdx * firstViewShape;
                        auto fullOffset1 = sIdx * secondViewShape;
                        auto fullOffset2 = mIdx * thirdViewShape;
                        auto fullOffset3 = nIdx * fourthViewShape;

                        std::vector<int64_t> baseView = {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape};
                        std::vector<SymbolicScalar> baseSize = {fullSize0, fullSize1, fullSize2, fullSize3};
                        std::vector<SymbolicScalar> baseOffset = {fullOffset0, fullOffset1, fullOffset2, fullOffset3};
                        
                        Tensor tileTensor0, tileTensor1;
                        if (diffCount == 1 && broadcastDim != -1) {
                            tileTensor0 = CreateViewForDim(inputs[0], baseView, baseSize, baseOffset,
                                                          inputs[0].GetShape()[broadcastDim] == 1 ? broadcastDim : -1);
                            tileTensor1 = CreateViewForDim(inputs[1], baseView, baseSize, baseOffset,
                                                          inputs[1].GetShape()[broadcastDim] == 1 ? broadcastDim : -1);
                        } else {
                            tileTensor0 = View(inputs[0], baseView, baseSize, baseOffset);
                            tileTensor1 = View(inputs[1], baseView, baseSize, baseOffset);
                        }

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = BitwiseXor(tileTensor0, tileTensor1);
                        Assemble(res, {fullOffset0, fullOffset1, fullOffset2, fullOffset3}, outputs[0]);
                    }
                }
            }
        }
    }
}

class BitwiseXorOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<BitwiseXorOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestBitwiseXor, BitwiseXorOperationTest,
    ::testing::ValuesIn(GetOpMetaData<BitwiseXorOpMetaData>(
        {BitwiseXorOperationExeFunc2Dims, BitwiseXorOperationExeFunc3Dims, BitwiseXorOperationExeFunc4Dims}, "BitwiseXor")));

TEST_P(BitwiseXorOperationTest, TestBitwiseXor) {
    auto test_data = GetParam().test_data_;
    auto args = BitwiseXorOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<BitwiseXorOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace