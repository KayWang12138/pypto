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
 * \file test_cast_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace ascend::test_operation;
namespace {
struct CastOpFuncArgs : public OpFuncArgs {
    CastOpFuncArgs(
        std::vector<int> shape, std::vector<int> vecTileShapes, DataType dTypeIn, DataType dTypeOut, CastMode castMode)
        : shape_(shape), vecTileShapes_(vecTileShapes), dTypeIn_(dTypeIn), dTypeOut_(dTypeOut), castMode_(castMode) {}
    std::vector<int> shape_;
    std::vector<int> vecTileShapes_;
    DataType dTypeIn_;
    DataType dTypeOut_;
    CastMode castMode_;
};
 
 struct CastOperationMetadata {
     CastOperationMetadata(std::vector<int> shape, std::vector<int> vecTileShapes, DataType dTypeIn, DataType dTypeOut,
         CastMode castMode, OpFunc opFunc)
         : args_(shape, vecTileShapes, dTypeIn, dTypeOut, castMode), opFunc_(opFunc) {}

     CastOpFuncArgs args_;
     OpFunc opFunc_;
 };
 
 [[maybe_unused]]static void CastOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                 const OpFuncArgs* opArgs) {
     FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
         SymbolicScalar firstDim = inputs[0]->shape[0];
         SymbolicScalar secondDim = inputs[0]->shape[1];
         const int firstl0LoopLengthTile = 128;
         DataType castDType = outputs[0].GetDataType();

         LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDivSymbolicScalar(firstDim, firstl0LoopLengthTile), 1)) {
             auto tileTensor0 = DViewPad(inputs[0], {firstl0LoopLengthTile, secondDim},
                 {std::min(firstDim - bIdx * firstl0LoopLengthTile, firstl0LoopLengthTile), secondDim},
                 {bIdx * firstl0LoopLengthTile, 0});
             Program::GetInstance().GetTileShape().SetVecTileShapes(
                 (static_cast<const CastOpFuncArgs*>(opArgs))->vecTileShapes_);
             auto res = Cast(tileTensor0, castDType, (static_cast<const CastOpFuncArgs*>(opArgs))->castMode_);
             DAssemble(res, {bIdx * firstl0LoopLengthTile, 0}, outputs[0]);
         }
     }
 }
 
 static void CastOperationExeFuncDoubleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs) {
     FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
         SymbolicScalar firstDim = inputs[0]->shape[0];
         SymbolicScalar secondDim = inputs[0]->shape[1];
         const int firstViewShape = 128;
         const int secondViewShape = 128;
         DataType castDType = outputs[0].GetDataType();

         LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx,
             LoopRange(0, CeilDivSymbolicScalar(firstDim, firstViewShape), 1)) {
             LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDivSymbolicScalar(secondDim, secondViewShape), 1)) {
                 auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                     {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                         std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                     {bIdx * firstViewShape, sIdx * secondViewShape});
                 Program::GetInstance().GetTileShape().SetVecTileShapes(
                     (static_cast<const CastOpFuncArgs*>(opArgs))->vecTileShapes_);
                 auto res = Cast(tileTensor0, castDType, (static_cast<const CastOpFuncArgs*>(opArgs))->castMode_);
                 DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
             }
         }
     }
 }

 static const CastOperationMetadata testDataLists[] = {
     CastOperationMetadata({512, 128}, {64, 128}, DataType::DT_FP32, DataType::DT_FP16, CAST_NONE, CastOperationExeFunc),
     CastOperationMetadata({1024, 256}, {64, 128}, DataType::DT_FP32, DataType::DT_BF16, CAST_NONE, CastOperationExeFunc),
     CastOperationMetadata({512, 256}, {64, 64}, DataType::DT_FP32, DataType::DT_INT16, CAST_RINT, CastOperationExeFuncDoubleCut),
     CastOperationMetadata({512, 128}, {64, 64}, DataType::DT_FP16, DataType::DT_FP32, CAST_NONE, CastOperationExeFuncDoubleCut),
     CastOperationMetadata({32, 32}, {32, 32}, DataType::DT_FP16,  DataType::DT_INT32, CAST_ROUND, CastOperationExeFuncDoubleCut),
     CastOperationMetadata({64, 64}, {32, 32}, DataType::DT_FP16,  DataType::DT_INT8, CAST_FLOOR, CastOperationExeFuncDoubleCut),
     CastOperationMetadata({90 + 17, 128 + 17}, {16, 16}, DataType::DT_INT32,  DataType::DT_FP16, CAST_NONE, CastOperationExeFuncDoubleCut),
     CastOperationMetadata({1, 1}, {16, 16}, DataType::DT_INT32, DataType::DT_FP32, CAST_NONE, CastOperationExeFuncDoubleCut),
     CastOperationMetadata({32, 32}, {16, 16}, DataType::DT_INT16, DataType::DT_FP32, CAST_NONE, CastOperationExeFuncDoubleCut),
     CastOperationMetadata({128, 128}, {64, 64}, DataType::DT_INT8, DataType::DT_FP16, CAST_NONE, CastOperationExeFuncDoubleCut),
 };

 class CastOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<CastOperationMetadata> {};
 
 INSTANTIATE_TEST_SUITE_P(
     TestCast,
     CastOperationTest,
     ::testing::ValuesIn(testDataLists)
 );
 
 TEST_P(CastOperationTest, test_cast) {
     TestCaseDesc testCase;
     testCase.inputTensors = {
         Tensor(GetParam().args_.dTypeIn_, GetParam().args_.shape_, "input0")
     };
     testCase.outputTensors = {
         Tensor(GetParam().args_.dTypeOut_, GetParam().args_.shape_, "output")
     };
     testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
     testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
     testCase.args = &(GetParam().args_);
     testCase.opFunc = GetParam().opFunc_;
     TestExecutor::runTest(testCase);
 }
 } // namespace