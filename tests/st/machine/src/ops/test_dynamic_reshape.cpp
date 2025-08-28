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
#include "test_dev_func_runner.h"

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
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

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
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

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
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), FuncRunnerConfig(q_real->GetDataSize()));

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
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), FuncRunnerConfig(q_real->GetDataSize()));

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
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

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
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(128*64, 1.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

/* 
    * test infershape case
*/

// test reshape unaligned infershape
TEST_F(DynamicReshapeTest, test_reshape_unalign) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 2;
    int sq = 64;
    int d = 64;
    std::vector<int> qShape2Dim = {b*sq, d};
    std::vector<int> qShape3Dim = {b, sq, d};


    Tensor q(DT_FP32, qShape2Dim, "q");
    Tensor actSeqs(DT_INT32, {b, 1, 1}, "actual_seq");
    Tensor out(DT_FP32, qShape3Dim, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {q, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0) / (sq))) {
            SymbolicScalar curSeq = GetInputDataInt32Dim3(actSeqs, batchId, 0, 0);

            Tensor q0 = DViewPad(q, {sq, d}, {curSeq, d}, {batchId * sq, 0});
            auto tmp0 = Reshape(q0, {1, sq, d}, {1, curSeq, d});
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 64, 64);
            auto tmp = Exp(tmp0);
            DAssemble(tmp, {batchId, 0, 0}, out);
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
    for (int bIdx = 0; bIdx < b; ++bIdx) {
        int offset = bIdx * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[bIdx] * d, exp(inputValue));
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

// test vec + mm diff tile and unaligned  infershape
TEST_F(DynamicReshapeTest, test_assemble_diff_tile) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {128, 128}, {128, 128});

    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int batch = 2;
    int s1 = 16;
    int s2 = 128;
    int d = 128;

    // (a + b)@c -> out  
    Tensor a(DT_FP32, {batch*s1, s2}, "a");
    Tensor b(DT_FP32, {batch*s2, d}, "b");
    Tensor out(DT_FP32, {batch*s1, d}, "out");

    Tensor actSeqs(DT_INT32, {batch}, "actual_seq");

    FUNCTION("main", FunctionType::DYNAMIC, {a, b, actSeqs}, {out}) {
        LOOP("LOOP_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(GetInputShapeDim(a, 0) / s1)) {
            SymbolicScalar actS2 = GetInputDataInt32Dim1(actSeqs, bIdx);

            Tensor aView = DViewPad(a, {s1, s2}, {s1, s2}, {bIdx*s1, 0});
            Tensor bView = DViewPad(b, {s2, d}, {s2, actS2}, {bIdx*s2, 0});

            Program::GetInstance().GetTileShape().SetVecTileShapes(16, 64);
            Tensor aFp16 = Cast(aView, DataType::DT_FP16);
            Program::GetInstance().GetTileShape().SetVecTileShapes(128, 64);
            Tensor bFp16 = Cast(bView, DataType::DT_FP16);

            auto tmpO = Matrix::Matmul<false, false>(DataType::DT_FP32, aFp16, bFp16);     // {s1, actS2} @ {actS2, d}
            DAssemble(tmpO, {bIdx*s1, 0}, out);
        }
    }

    float inputValue = 1.0f;
    float initValue = 0.5f;
    int acutalValue = 62;
    std::vector<int> actSeqsData(batch, acutalValue);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(a, inputValue),
        RawTensorData::CreateConstantTensor<float>(b, inputValue),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(batch * s1 * d, initValue);
    for (int bsIdx = 0; bsIdx < batch * s1; ++bsIdx) {
        int offset = bsIdx * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + acutalValue, 128.0f);
    }
    
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

// test DView + Reshape + DAssemble 4->2 + op  2batch will wrong
TEST_F(DynamicReshapeTest, test_reshape_dassemble_4_2) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 64, 64);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 2;
    int s = 1;
    int n1 = 64;
    int d = 64;

    // [b,s1,n1,d] -> [b*s1*n1,d]
    Tensor queryOut(DT_FP32, {b, s, n1, d}, "queryOut");
    Tensor qNope(DT_FP32, {b * s * n1, d}, "qNope");
    Tensor qRes(DT_FP32, {b * s * n1, d}, "qRes");

    FUNCTION("main", FunctionType::DYNAMIC, {queryOut}, {qNope, qRes}) {
        LOOP("RESHAPE_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b, 1), {}, true) {
            SymbolicScalar bOffset = bIdx * 1;
            LOOP("RESHAPE_LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, s, 1)) {
                SymbolicScalar sOffset = sIdx * 1;

                Tensor nopeView = DView(queryOut, {1, 1, n1, d}, {bOffset, sOffset, 0, 0});
                Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, 32, d});
                Tensor nopeRes = Reshape(nopeView, {1 * 1 * n1, d});
                DAssemble(nopeRes, {(bOffset * s + sOffset) * n1, 0}, qNope);
            }
        }

        LOOP("Add_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b, 1), {}, true) {
            auto qNopeL = DView(qNope, {64, 64}, {bIdx * s * n1, 0});
            Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
            auto qResTmp = AddS(qNopeL, Element(DataType::DT_FP32, 1.0));
            DAssemble(qResTmp, {bIdx * s * n1, 0}, qRes);
        }
    }

    float inputValue = 2.0f;
    float initValue = 0.5f;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(queryOut, inputValue),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(qNope, initValue),
        RawTensorData::CreateConstantTensor<float>(qRes, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden_qNope(b * s * n1 * d, inputValue);
    std::vector<float> golden_qRes(b * s * n1 * d, inputValue + 1.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputDataList();
    EXPECT_TRUE(resultCmp(golden_qNope, (float *)outs[0]->data(), 0.001f)); //right
    EXPECT_TRUE(resultCmp(golden_qRes, (float *)outs[1]->data(), 0.001f));  //wrong

}

//  dassemble + op + unaligin  Dassemble 不推导 validshape而是使用dst的shape时，后续操作会有问题

/* 
    * test copy case
*/

// test DView + Reshape + DAssemble 2->3
TEST_F(DynamicReshapeTest, test_reshape_dassemble) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 1;
    int sq = 64;
    int d = 64;
    std::vector<int> qShape2Dim = {b*sq, d};
    std::vector<int> qShape3Dim = {b, sq, d};


    Tensor q(DT_FP32, qShape2Dim, "q");
    Tensor out(DT_FP32, qShape3Dim, "out");

# if 1
    FUNCTION("main", FunctionType::DYNAMIC, {q}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0) / (sq))) {
            Tensor q0 = DView(q, {sq, d}, {batchId * sq, 0});
            // auto tmp0 = MulS(q0, Element(DataType::DT_FP32, 1.0));
            auto tmp = Reshape(q0, {1, sq, d});
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 64, 64);
            // auto tmp = MulS(tmp, Element(DataType::DT_FP32, 1.0));
            DAssemble(tmp, {batchId, 0, 0}, out);
        }
    }
#else
    FUNCTION("main", FunctionType::DYNAMIC, {q}, {out}) {
        Tensor q0 = DView(q, {sq, d}, {sq, 0});
        auto tmp0 = MulS(q0, Element(DataType::DT_FP32, 1.0));
        auto tmp = Reshape(q0, {1, sq, d});
        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 64, 64);
        DAssemble(tmp, {0, 0, 0}, out);
    }
#endif

    float inputValue = 2.0f;
    float initValue = 0.5f;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, inputValue),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(b * sq * d, inputValue);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}


// ===================  reshape + op + reshape  ??????
TEST_F(DynamicReshapeTest, test_reshape_op_reshape) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 64, 64);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 2;
    int s = 1;
    int n1 = 64;
    int d = 64;

     // [b,s1,n1,d] -> [b*s1*n1,d]
    Tensor queryOut(DT_FP32, {b, s, n1, d}, "queryOut");
    Tensor qNope(DT_FP32, {b * s, n1, d}, "qNope");

    FUNCTION("main", FunctionType::DYNAMIC, {queryOut}, {qNope}) {
        LOOP("RESHAPE_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b, 1), {}, true) {
            SymbolicScalar bOffset = bIdx * 1;
            LOOP("RESHAPE_LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, s, 1)) {
                SymbolicScalar sOffset = sIdx * 1;
                Tensor nopeView = DView(queryOut, {1, 1, n1, d}, {bOffset, sOffset, 0, 0});
                Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, 64, 64});
                Tensor tmp0 = Reshape(nopeView, {1 * 1 * n1, d});
                auto tmp1 = AddS(tmp0, Element(DataType::DT_FP32, 1.0));
                Program::GetInstance().GetTileShape().SetVecTileShapes({1, 64, 64});
                auto tmp2 = Reshape(tmp1, {1, n1, d});
                auto nopeRes = MulS(tmp2, Element(DataType::DT_FP32, 1.0));
                DAssemble(nopeRes, {(bOffset * s + sOffset), 0, 0}, qNope);
            }
        }
    }

    float inputValue = 2.0f;
    float initValue = 0.5f;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(queryOut, inputValue),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(qNope, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden_qNope(b * s * n1 * d, inputValue + 2.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputDataList();
    EXPECT_TRUE(resultCmp(golden_qNope, (float *)outs[0]->data(), 0.001f)); //right
}
