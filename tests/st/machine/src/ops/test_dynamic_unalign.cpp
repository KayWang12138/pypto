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
 * \file test_dynamic_unalign.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "operation/tilefwk_op.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dynamic.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicUnalignTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        config::SetHostConfig(KEY_ONLY_CODEGEN, true);
        Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
        Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});
    }
};

void TestLoopTailBlock(const Tensor &t0, const Tensor &blockTable, Tensor &out, int s) {
    int blockSize = 64;
    FUNCTION("main", FunctionType::DYNAMIC, {t0, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShapeDim(t0, 0) / s)) {
            SymbolicScalar size = GetInputDataInt32Dim2(blockTable, i, 0);
            Tensor t0s = DViewPad(t0, {s, s}, {size, s}, {blockSize * i, 0});
            Tensor t1 = Add(t0s, t0s);
            DAssemble(t1, {blockSize * i, 0}, out);
        }
    }
}

TEST_F(DynamicUnalignTest, TestTailBlock) {
    int s = 64;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
    Tensor out(DT_FP32, {n * s, s}, "out");
    TestLoopTailBlock(t0, blockTable, out, s);

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData),  // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcop);
    std::vector<float> golden(n * s * s, 2.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicUnalignTest, test_mm_unalign) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 1;
    int nq = 32;
    int nk = 1;
    int s1 = 1;
    int s2 = 256;
    int dR = 64;
    int dN = 512;

    std::vector<int> qRopeShape = {b * nq * s1, dR};
    std::vector<int> qNopeShape = {b * nq * s1, dN};

    std::vector<int> kRopeShape = {b * nk * s2, dR};
    std::vector<int> kNopeShape = {b * nk * s2, dN};

    std::vector<int> outShape = {b * nq * s1, nk * s2};

    Tensor qRope(DT_BF16, qRopeShape, "qRope");
    Tensor qNope(DT_BF16, qNopeShape, "qNope");

    Tensor kRope(DT_BF16, kRopeShape, "kRope");
    Tensor kNope(DT_BF16, kNopeShape, "kNope");

    Tensor actSeqs{DT_INT32, {b}, "actSeqs"};
    Tensor out(DT_FP32, outShape, "out");

    auto dtype = qNope->Datatype();
    FUNCTION("main", FunctionType::DYNAMIC, {qRope, qNope, kRope, kNope, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(qRope, 0) / (nq * s1))) {
            SymbolicScalar curSeq = GetInputDataInt32Dim1(actSeqs, batchId);

            Tensor qr = DViewPad(qRope, {nq * s1, dR}, {nq * s1, dR}, {batchId * nq * s1, 0});
            Tensor qn = DViewPad(qNope, {nq * s1, dN}, {nq * s1, dN}, {batchId * nq * s1, 0});

            Tensor kr = DViewPad(kRope, {nk * s2, dR}, {nk * curSeq, dR}, {batchId * nk * s2, 0});
            Tensor kn = DViewPad(kNope, {nk * s2, dN}, {nk * curSeq, dN}, {batchId * nk * s2, 0});

            Tensor qi(dtype, {nq * s1, dN + dR}, "qi");
            DAssemble(qn, {0, 0}, qi);
            DAssemble(qr, {0, dN}, qi);

            Tensor kj(dtype, {nk * s2, dN + dR}, "kj");
            DAssemble(kn, {0, 0}, kj);
            DAssemble(kr, {0, dN}, kj);

            auto tmp = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj);

            DAssemble(tmp, {batchId * nq * s1, 0}, out);
        }
    }

    // read data
    std::vector<npu::tile_fwk::bfloat16> qRopeData(b * nq * s1 * dR, 0);
    std::vector<npu::tile_fwk::bfloat16> qNopeData(b * nq * s1 * dN, 0);

    std::vector<npu::tile_fwk::bfloat16> kRopeData(b * nk * s2 * dR, 0);
    std::vector<npu::tile_fwk::bfloat16> kNopeData(b * nk * s2 * dN, 0);

    std::vector<int> seq(b);
    std::vector<float> golden(b * nq * s1 * nk * s2, 0);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_rope.bin", qRopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_nope.bin", qNopeData);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_rope.bin", kRopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_nope.bin", kNopeData);

    readInput<int>(GetGoldenDir() + "/actual_seq_len.bin", seq);
    readInput(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qRope, qRopeData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qNope, qNopeData),

        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kRope, kRopeData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kNope, kNopeData),

        RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.005f));
// #endif
}

TEST_F(DynamicUnalignTest, test_mm2_unalign) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {128, 128}, {64, 64});
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 1;
    int nq = 32;
    int nk = 1;
    int s1 = 1;
    int s2 = 128;
    int d = 64;

    std::vector<int> qShape = {b * nq * s1, nk * s2}; // {32, 128}
    std::vector<int> kShape = {b * nk * s2, d}; // 128 64

    std::vector<int> outShape = {b * nq * s1, d};

    Tensor qk(DT_BF16, qShape, "qk");
    Tensor v(DT_BF16, kShape, "v");
    Tensor actSeqs{DT_INT32, {b}, "actSeqs"};
    Tensor out(DT_FP32, outShape, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {qk, v, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(qk, 0) / (nq * s1))) {
            SymbolicScalar curSeq = GetInputDataInt32Dim1(actSeqs, batchId);

            Tensor qk0 = DViewPad(qk, {nq * s1, nk * s2}, {nq * s1, nk * curSeq}, {batchId * nq * s1, 0});
            Tensor v0 = DViewPad(v, {nk * s2, d}, {nk * curSeq, d}, {batchId * nk * s2, 0});
            auto tmp = Matrix::Matmul<false, false>(DataType::DT_FP32, qk0, v0);

            DAssemble(tmp, {batchId * nq * s1, 0}, out);
            // out = Matrix::Matmul<false, false>(DataType::DT_FP32, qk0, v0);
        }
    }

    // read data
    std::vector<npu::tile_fwk::bfloat16> qkData(b * nq * s1 * nk * s2, 0);
    std::vector<npu::tile_fwk::bfloat16> vData(b * nk * s2 * d, 0);
    std::vector<int> seq(b);
    std::vector<float> golden(b * nq * s1 * d, 0);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/qk.bin", qkData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/v.bin", vData);
    readInput<int>(GetGoldenDir() + "/actual_seq_len.bin", seq);
    readInput(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qk, qkData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(v, vData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.005f));
// #endif
}

TEST_F(DynamicUnalignTest, test_rowmaxsingle_unalign) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(128, 128);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 1;
    int nTile = 32;
    int blockSize = 256;
    std::vector<int> qShape = {b * nTile , blockSize};
    std::vector<int> outshape = {b * nTile , 1};

    Tensor q(DT_FP32, qShape, "q");
    Tensor actSeqs(DT_INT32, {b, 1}, "actual_seq");
    Tensor out(DT_FP32, outshape, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {q, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0) / (nTile))) {
            SymbolicScalar curSeq = GetInputDataInt32Dim2(actSeqs, batchId, 0);

            Tensor q0 = DViewPad(q, {nTile, blockSize}, {nTile, curSeq}, {batchId * nTile, 0});
            auto tmp = RowMaxSingle(q0);
            DAssemble(tmp, {batchId * nTile, 0}, out);
        }
    }

    // read data
    std::vector<float> qData(b * nTile * blockSize, 0);

    std::vector<int> seq(b);
    std::vector<float> golden(b * nTile * 1, 0);

    readInput<float>(GetGoldenDir() + "/q.bin", qData);
    readInput<int>(GetGoldenDir() + "/actual_seq_len.bin", seq);
    readInput(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(q, qData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

TEST_F(DynamicUnalignTest, test_rowsumsingle_unalign) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(128, 128);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 1;
    int nTile = 32;
    int blockSize = 256;
    std::vector<int> qShape = {b * nTile, blockSize};
    std::vector<int> outshape = {b * nTile, 1};

    Tensor q(DT_FP32, qShape, "q");
    Tensor actSeqs(DT_INT32, {b, 1}, "actual_seq");
    Tensor out(DT_FP32, outshape, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {q, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0) / (nTile))) {
            SymbolicScalar curSeq = GetInputDataInt32Dim2(actSeqs, batchId, 0);

            Tensor q0 = DViewPad(q, {nTile, blockSize}, {nTile, curSeq}, {batchId * nTile, 0});
            auto tmp = RowSumSingle(q0, -1);
            DAssemble(tmp, {batchId * nTile, 0}, out);
        }
    }

    // read data
    std::vector<float> qData(b * nTile * blockSize, 0);

    std::vector<int> seq(b);
    std::vector<float> golden(b * nTile * 1, 0);

    readInput<float>(GetGoldenDir() + "/q.bin", qData);
    readInput<int>(GetGoldenDir() + "/actual_seq_len.bin", seq);
    readInput(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(q, qData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

TEST_F(DynamicUnalignTest, test_unary_unalign) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int b = 4;
    int sq = 128;
    int d = 64;
    std::vector<int> qShape = {b * sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor actSeqs(DT_INT32, {b, 1, 1}, "actual_seq");
    Tensor out(DT_FP32, qShape, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {q, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShapeDim(q, 0) / (sq))) {
            SymbolicScalar curSeq = GetInputDataInt32Dim3(actSeqs, batchId, 0, 0);

            Tensor q0 = DViewPad(q, {sq, d}, {curSeq, d}, {batchId * sq, 0});
            auto tmp = Exp(q0);
            DAssemble(tmp, {batchId * sq, 0}, out);
        }
    }

    std::vector<int> actSeqsData(b, 100);
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, 1.0),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    // excute
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);

    std::vector<float> golden(b * sq * d, 0.001f);
    for (int bidx = 0; bidx < b; ++bidx) {
        int offset = bidx * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[bidx] * d, exp(1.0));
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}
