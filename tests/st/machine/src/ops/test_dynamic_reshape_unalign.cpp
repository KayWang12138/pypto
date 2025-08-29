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
 * \file test_dynamic_reshape_unalign.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "tilefwk/function.h"
#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicReshapeUnalignTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    }
};

// add dim
TEST_F(DynamicReshapeUnalignTest, test_reshape_unalign_add_dim) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 2;
    int sq = 64;
    int d = 64;
    std::vector<int64_t> qShape2Dim = {b*sq, d};
    std::vector<int64_t> qShape3Dim = {b, sq, d};

    Tensor q(DT_FP32, qShape2Dim, "q");
    Tensor actSeqs(DT_INT32, {b, 1, 1}, "actual_seq");
    Tensor out(DT_FP32, qShape3Dim, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {q, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0) / (sq))) {
            SymbolicScalar curSeq = GetInputDataInt32Dim3(actSeqs, batchId, 0, 0);
            Tensor q0 = View(q, {sq, d}, {curSeq, d}, {batchId * sq, 0});
            auto tmp0 = Reshape(q0, {1, sq, d}, {1, curSeq, d});
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 64, 64);
            auto tmp = Exp(tmp0);
            Assemble(tmp, {batchId, 0, 0}, out);
        }
    }

    float inputValue = 2.0f;
    float initValue = 0.5f;

    std::vector<int> actSeqsData(b, 63);
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, inputValue),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(b * sq * d, initValue);
    for (int bidx = 0; bidx < b; ++bidx) {
        int offset = bidx * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[bidx] * d, exp(inputValue));
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

// merge dim, not last dim
TEST_F(DynamicReshapeUnalignTest, test_reshape_unalign_merge_dim) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 16, 16);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 2;
    int sq = 10;
    int d = 10;
    std::vector<int64_t> qShape3Dim = {b, sq, d};
    std::vector<int64_t> qShape2Dim = {b, sq * d};

    Tensor q(DT_FP32, qShape3Dim, "q");
    Tensor actSeqs(DT_INT32, {b, 1, 1}, "actual_seq");
    Tensor out(DT_FP32, qShape2Dim, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {q, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
            //
            SymbolicScalar curSeq = GetInputDataInt32Dim3(actSeqs, batchId, 0, 0);

            Tensor q0 = View(q, {1, sq, d}, {1, curSeq, d}, {batchId, 0, 0});
            auto tmp0 = Reshape(q0, {1, sq * d}, {1, curSeq * d});
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 16);
            auto tmp = Exp(tmp0);
            Assemble(tmp, {batchId, 0}, out);// 1, sq * d -> b, sq * d
        }
    }

    float inputValue = 2.0f;
    float initValue = 0.5f;

    std::vector<int> actSeqsData(b, 8);
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, inputValue),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(b * sq * d, initValue);
    for (int bidx = 0; bidx < b; ++bidx) {
        int offset = bidx * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[bidx] * d, exp(inputValue));
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

// split dim
TEST_F(DynamicReshapeUnalignTest, test_reshape_unalign_split_dim) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 16, 16);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 2;
    int sq = 6;
    int d = 10;
    std::vector<int64_t> qShape3Dim = {b, sq, d};
    std::vector<int64_t> qShape4Dim = {b, sq, 5, d/5};

    Tensor q(DT_FP32, qShape3Dim, "q");
    Tensor actSeqs(DT_INT32, {b, 2, 1}, "actual_seq");
    Tensor out(DT_FP32, qShape4Dim, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {q, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
            SymbolicScalar curSeq = GetInputDataInt32Dim3(actSeqs, batchId, 0, 0);
            SymbolicScalar curDim = GetInputDataInt32Dim3(actSeqs, batchId, 1, 0);
            Tensor q0 = View(q, {1, sq, d}, {1, curSeq, curDim}, {batchId, 0, 0});
            auto tmp0 = Reshape(q0, {1, sq, 5, d/5}, {1, curSeq, 4, curDim/4});
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 16, 16, 16);

            auto tmp = MulS(tmp0, Element(tmp0->Datatype(), 1.0));
            Assemble(tmp, {batchId, 0, 0, 0}, out);// 1, sq * d -> b, sq * d
        }
    }

    float initValue = 0.5f;

    std::vector<int> actSeqsData = {5, 8, 5, 8};
    std::vector<float> inputValueData;
    for (int i = 0; i < b * sq * d; i++){
        inputValueData.push_back(static_cast<float>(i));
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(q, inputValueData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(b * sq * d, initValue);
    int count = 0;
    for (int bidx = 0; bidx < b; ++bidx) {
        int offset = bidx * sq * d;
        count = offset;
        for (int row = 0; row < actSeqsData[0]; row++){
            for (int col = 0; col < actSeqsData[1]; col++){
                if (count % d == 8) count +=2;
                golden[offset + row * d + col] = count++;
            }
        }
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

// split dim and merge dim
TEST_F(DynamicReshapeUnalignTest, test_reshape_unalign_split_and_merge) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 4, 32);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetPassConfig("PVC2_OOO", "SplitReshape", "DISABLE_PASS", true);

    int b = 4;
    int sq = 128;
    int d = 64;
    std::vector<int64_t> qShape3Dim = {b, sq, d};
    Tensor q(DT_FP32, qShape3Dim, "q");
    Tensor out(DT_FP32, qShape3Dim, "out");

    int bView = 2;
    int sqView = 12;
    int dView = 64;

    FUNCTION("main", FunctionType::DYNAMIC, {q}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, (b + bView - 1) / bView, 1)) {
            SymbolicScalar bValid = min(b - bView * bIdx, bView);
            LOOP("L1", FunctionType::DYNAMIC_LOOP, sqIdx, LoopRange(0, (sq + sqView - 1) / sqView, 1)) {
                SymbolicScalar sqValid = min(sq - sqView * sqIdx, sqView);
                Tensor q0 = View(q, {bView, sqView, dView}, {bValid, sqValid, dView}, {bIdx * bView, sqIdx * sqView, 0});
                Tensor tmp0 = Reshape(q0, {bView * sqView, dView}, {bValid * sqValid, dView}); //(bView, sqView, dView) -> (bView * sqView, dView)

                Program::GetInstance().GetTileShape().SetVecTileShapes(1*4, 32);
                Tensor tmp1 = AddS(tmp0, Element(tmp0->Datatype(), 0.01));
                Tensor tmp2 = Reshape(tmp1, {bView, sqView, dView}, {bValid, sqValid, dView}); //(bView * sqView, dView) -> (bView, sqView, dView)
                Program::GetInstance().GetTileShape().SetVecTileShapes(1, 4, 32);
                Assemble(tmp2, {bIdx * bView, sqIdx * sqView, 0}, out);
            }
        }
    }
    float initInputValue = 2.0f;
    float initOutValue = 0.5f;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, initInputValue),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initOutValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(b * sq * d, 2.01f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}