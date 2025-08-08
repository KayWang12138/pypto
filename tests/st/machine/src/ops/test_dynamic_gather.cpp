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
 * \file test_dynamic_gather.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/page_attention.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "runtime.h"
#include "device_runner.h"
#include "cost_model/simulation/backend.h"
#include "test_dynamic.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicGatherTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};
TEST_F(DynamicGatherTest, TestDynamicGatherDim2) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int s1 = 128;
    int b = 2;
    int sq = 128;
    int d = 64;
    std::vector<int> qShape = {s1, d};
    std::vector<int> indicesShape = {b * sq};
    std::vector<int> outShape = {b * sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor indices(DT_INT32, indicesShape, "indices");
    Tensor actSeqs(DT_INT32, {b, 1}, "actual_seq");
    Tensor out(DT_FP32, outShape, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {q, indices, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
            int axis = 0;
            SymbolicScalar curSeq = GetInputDataInt32Dim2(actSeqs, batchId, 0);

            Tensor indices0 = DViewPad(indices, {sq}, {curSeq}, {batchId * sq});
            auto tmp = Gather(q, indices0, axis);
            DAssemble(tmp, {batchId * sq, 0}, out);
        }
    }

    std::vector<int> actSeqsData(b, 100);
    std::vector<float> qData(s1 * d, 0);
    std::vector<float> golden(b * sq * d, 0);
    std::vector<int> indicesData(b * sq, 0);
    readInput<float>(GetGoldenDir() + "/x.bin", qData);
    readInput<int>(GetGoldenDir() + "/indices.bin", indicesData);
    readInput<float>(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(q, qData),
        RawTensorData::CreateTensor<int>(indices, indicesData),
        RawTensorData::CreateTensor<int>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

TEST_F(DynamicGatherTest, TestDynamicGatherDim3) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 64);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int s1 = 32;
    int b = 2;
    int sq = 32;
    int d = 64;
    int s2 = 2;
    std::vector<int> qShape = {s1, d};
    std::vector<int> indicesShape = {b * sq, s2};
    std::vector<int> outShape = {b * sq, s2, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor indices(DT_INT32, indicesShape, "indices");
    Tensor actSeqs(DT_INT32, {b, 1}, "actual_seq");
    Tensor out(DT_FP32, outShape, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {q, indices, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
            int axis = 0;
            SymbolicScalar curSeq = GetInputDataInt32Dim2(actSeqs, batchId, 0);

            Tensor indices0 = DViewPad(indices, {sq, s2}, {curSeq, s2}, {batchId * sq, 0});
            auto tmp = Gather(q, indices0, axis);
            DAssemble(tmp, {batchId * sq, 0, 0}, out);
        }
    }

    std::vector<int> actSeqsData(b, 30);
    std::vector<float> qData(s1 * d, 0);
    std::vector<float> golden(b * sq * s2 * d, 0);
    std::vector<int> indicesData(b * sq * s2, 0);
    readInput<float>(GetGoldenDir() + "/x.bin", qData);
    readInput<int>(GetGoldenDir() + "/indices.bin", indicesData);
    readInput<float>(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(q, qData),
        RawTensorData::CreateTensor<int>(indices, indicesData),
        RawTensorData::CreateTensor<int>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}
