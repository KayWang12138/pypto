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
 * \file test_dynamic_binry_brc.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "models/deepseek/page_attention.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dynamic.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicBrcTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(DynamicBrcTest, TestDynamicMulBrcUnalign) {
    config::SetOperationConfig("FORCE_COMBINE_AXIS", true);
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 128);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 2;
    int sq = 32;
    int d = 72;
    std::vector<int> inputShape_a = {b * sq, d};
    std::vector<int> inputShape_b = {b * sq, 8};
    std::vector<int> outShape = {b * sq, d};

    Tensor input_a(DT_FP32, inputShape_a, "intput_a");
    Tensor input_b(DT_FP32, inputShape_b, "intput_b");
    Tensor curSeq(DT_FP32, {b, 1}, "curSeq");
    Tensor out(DT_FP32, outShape, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {input_a, input_b, curSeq}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
            auto seq = GetInputDataInt32Dim2(curSeq, batchId, 0);
            Tensor input_a0 = DViewPad(input_a, {sq, d}, {seq, d}, {batchId * sq, 0});
            Tensor input_b0 = DViewPad(input_b, {sq, 8}, {seq, 8}, {batchId * sq, 0});
            auto input_c = RowSumSingle(input_b0);
            auto tmp = Mul(input_a0, input_c);
            DAssemble(tmp, {batchId * sq, 0}, out);
        }
    }

    std::vector<int> actSeqsData(b, 24); // 有效值应该是block对齐的

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input_a, 2.0),
        RawTensorData::CreateConstantTensor<float>(input_b, 3.0),
        RawTensorData::CreateTensor<int32_t>(curSeq, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    std::vector<float> golden(b * sq * d, 0.001f);
    for (int i = 0; i < b; i++) {
        int offset = i * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[i] * d, 48.0);
    }
    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}
