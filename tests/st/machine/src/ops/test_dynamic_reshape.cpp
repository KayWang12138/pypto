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
 * \file test_dynamic_reshape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/function.h"
#include "test_suite_stest_ops.h"
#include "test_dynamic.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicReshapeTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    }
};

TEST_F(DynamicReshapeTest, test_only_reshape) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 1;
    int sq = 128;
    int d = 64;
    int bSq = (b == -1) ? -1 : b*sq;
    std::vector<int> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, {bSq, d}, "out");

    FUNCTION("MAIN_FUNC", FunctionType::DYNAMIC, {q}, {out}) {
        Tensor bfRes(DT_FP32, qShape, "bfRes");
        LOOP("L0_BF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0)), {}, true) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 64, 64);
            Tensor q0 = DView(q, {1, sq, d}, {batchId, 0, 0});
            auto tmp = Exp(q0);
            DAssemble(tmp, {batchId, 0, 0}, bfRes);
        }

        Tensor qReshape(DT_FP32, {bSq, d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0,1,1), {}, true) {
            (void) batchId;
            ReshapeInplace(bfRes, qReshape);
        }

        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0)), {}, true) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
            Tensor q0 = DView(qReshape, {sq, d}, {batchId * sq, 0});
            auto tmp = AddS((q0), Element(DataType::DT_FP32, 1.0f));
            DAssemble(tmp, {batchId * sq, 0}, out);
        }
    }

    b = 1;
    Tensor q_real(DT_FP32, {b, sq, d});
    Tensor out_real(DT_FP32, {b * sq, d});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q_real, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out_real, 0.001f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);

    std::vector<float> golden(b * sq * d, exp(1.0f) + 1.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

TEST_F(DynamicReshapeTest, test_only_reshape2) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 1;
    int sq = 128;
    int d = 64;
    int bSq = (b == -1) ? -1 : b*sq;
    std::vector<int> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, {bSq, d}, "out");

    FUNCTION("MAIN_FUNC", FunctionType::DYNAMIC, {q}, {out}) {
        Tensor qReshape(DT_FP32, {bSq, d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0,1,1), {}, true) {
            (void) batchId;
            ReshapeInplace(q, qReshape);
        }

        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0)), {}, true) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
            Tensor q0 = DView(qReshape, {sq, d}, {batchId * sq, 0});
            auto tmp = Exp(q0);
            DAssemble(tmp, {batchId * sq, 0}, out);
        }
    }

    b = 1;
    Tensor q_real(DT_FP32, {b, sq, d});
    Tensor out_real(DT_FP32, {b * sq, d});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q_real, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out_real, 0.001f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);

    std::vector<float> golden(b * sq * d, exp(1.0f));

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

TEST_F(DynamicReshapeTest, test_dyn_reshape) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = -1;
    int sq = 128;
    int d = 64;
    int bSq = (b == -1) ? -1 : b*sq;
    std::vector<int> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, {bSq, d}, "out");

    FUNCTION("MAIN_FUNC", FunctionType::DYNAMIC, {q}, {out}) {
        Tensor qReshape(DT_FP32, {GetInputShapeDim(q, 0) * GetInputShapeDim(q, 1), d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0,1,1), {}, true) {
            (void) batchId;
            ReshapeInplace(q, qReshape);
        }

        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0)), {}, true) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
            Tensor q0 = DView(qReshape, {sq, d}, {batchId * sq, 0});
            auto tmp = Exp(q0);
            DAssemble(tmp, {batchId * sq, 0}, out);
        }
    }

    b = 1;
    Tensor q_real(DT_FP32, {b, sq, d});
    Tensor out_real(DT_FP32, {b * sq, d});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q_real, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out_real, 0.001f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop, DynFuncRunnerConfig(q_real->GetDataSize()));

    std::vector<float> golden(b * sq * d, exp(1.0f));

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

TEST_F(DynamicReshapeTest, test_dyn_reshape2) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = -1;
    int sq = 128;
    int d = 64;
    int bSq = (b == -1) ? -1 : b*sq;
    std::vector<int> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, {bSq, d}, "out");

    FUNCTION("MAIN_FUNC", FunctionType::DYNAMIC, {q}, {out}) {
        Tensor bfRes(DT_FP32,  {GetInputShapeDim(q, 0), sq, d}, "bfRes");
        LOOP("L0_BF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0)), {}, true) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 64, 64);
            Tensor q0 = DView(q, {1, sq, d}, {batchId, 0, 0});
            auto tmp = Exp(q0);
            DAssemble(tmp, {batchId, 0, 0}, bfRes);
        }

        Tensor qReshape(DT_FP32, {GetInputShapeDim(q, 0) * GetInputShapeDim(q, 1), d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0,1,1), {}, true) {
            (void) batchId;
            ReshapeInplace(bfRes, qReshape);
        }

        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0)), {}, true) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
            Tensor q0 = DView(qReshape, {sq, d}, {batchId * sq, 0});
            auto tmp = AddS((q0), Element(DataType::DT_FP32, 1.0f));
            DAssemble(tmp, {batchId * sq, 0}, out);
        }
    }

    b = 1;
    Tensor q_real(DT_FP32, {b, sq, d});
    Tensor out_real(DT_FP32, {b * sq, d});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q_real, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out_real, 0.001f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop, DynFuncRunnerConfig(q_real->GetDataSize()));

    std::vector<float> golden(b * sq * d, exp(1.0f) + 1.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

TEST_F(DynamicReshapeTest, test_dyn_reshape1111) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 64);

    Tensor A(DT_FP32, {128, 64}, "A");
    Tensor B(DT_FP32, {128, 64}, "B");
    Tensor D(DT_FP32, {256, 64}, "D");

    FUNCTION("MAIN_FUNC", FunctionType::DYNAMIC, {A, B}, {D}) {
        LOOP("LOOP_TEST", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange(0,2,1)) {
            Tensor C(DT_FP32, {128, 64}, "q");
            auto a0 = DView(A, {64, 64}, {loopIdx * 64, 0});
            auto a1 = AddS(a0, Element(DataType::DT_FP32, 1.0f));
            DAssemble(a1, {0, 0}, C);

            auto b0 = DView(B, {64, 64}, {loopIdx * 64, 0});
            auto b1 = AddS(b0, Element(DataType::DT_FP32, 1.0f));
            DAssemble(b1, {64, 0}, C);

            auto d = AddS(C, Element(DataType::DT_FP32, 1.0f));
            DAssemble(d, {loopIdx * 128, 0}, D);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(A, 1.0),
        RawTensorData::CreateConstantTensor<float>(B, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(D, 0.001f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);

    std::vector<float> golden(256*64, 3.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}


TEST_F(DynamicReshapeTest, test_dyn_reshape22222) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 64);

    Tensor A(DT_FP32, {128, 64}, "A");
    Tensor B(DT_FP32, {128, 64}, "B");

    FUNCTION("MAIN_FUNC", FunctionType::DYNAMIC, {A}, {B}) {
        LOOP("LOOP_TEST", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange(0,1,1)) {
            (void) loopIdx;
            DAssemble(A, {0, 0}, B);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(A, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(B, 0.001f),

    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);

    std::vector<float> golden(128*64, 1.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}