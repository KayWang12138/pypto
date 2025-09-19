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
 * \file test_dynamic_reshape_unalign_inplace.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "tilefwk/function.h"
#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicReshapeUnalignImplaceTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    }
};

void genDateAndExe(Tensor in, Tensor out, int opCount){
    std::vector<int64_t> shape = in.GetShape();
    size_t elementSum = 1;
    for (size_t i = 0; i < shape.size(); i++){
        elementSum *= shape[i];
    }

    std::vector<float> inputValueData(elementSum, 0);
    for (size_t i = 0; i < elementSum ; i++){
        inputValueData[i] = static_cast<float>(i);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(in, inputValueData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    std::vector<float> golden(elementSum, 0);
    for (size_t i = 0; i < elementSum; i++){
        golden[i] = static_cast<float>(i) + 0.01f * opCount;
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), DeviceLauncherConfig(in->GetDataSize()));
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

TEST_F(DynamicReshapeUnalignImplaceTest, merge_two_dynamic_dim) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    int b = -1;
    int sq = -1;
    int d = 64;
    int bSq = (b == -1) ? -1 : b*sq;
    std::vector<int64_t> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");        // (-1(2), -1(3), 64)
    Tensor out(DT_FP32, {bSq, d}, "out");  // (-1(2*3), 64)

    FunctionConfig funConfig;
    FUNCTION("MAIN_FUNC", funConfig, {q}, {out}) {
        Tensor qReshape(DT_FP32, {GetInputShape(q, 0) * GetInputShape(q, 1), d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0,1,1), {}, true) {
            (void) batchId;
            ReshapeInplace(q, qReshape);
        }
        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0)), {}, true) {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, sqId, LoopRange(GetInputShape(q, 1)), {}, true) {
                TileShape::Current().SetVecTile(64, 64);
                Tensor q0 = View(qReshape, {1, d}, {batchId * GetInputShape(q, 1) + sqId, 0});
                Tensor tmp = AddS(q0, Element(q0->Datatype(), 0.01));
                Assemble(tmp, {batchId * GetInputShape(q, 1) + sqId, 0}, out);
            }
        }
    }
    b = 2;
    sq = 3;
    Tensor q_real(DT_FP32, {b, sq, d});
    Tensor out_real(DT_FP32, {b * sq, d});
    genDateAndExe(q_real, out_real, 1);
}

TEST_F(DynamicReshapeUnalignImplaceTest, test_exchange_dim) {
    TileShape::Current().SetVecTile(1, 16, 16);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int sq = 12;
    int d = -1;
    int m = 8;
    std::vector<int64_t> qShape3Dim = {sq, d, m}; // (12, -1(24), 8)
    std::vector<int64_t> qShape2Dim = {d, sq, m}; // (-1(24), 12, 8)

    Tensor q(DT_FP32, qShape3Dim, "q");
    Tensor out(DT_FP32, qShape2Dim, "out");

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {q}, {out}) {
        Tensor q_reshape(DT_FP32, {GetInputShape(q, 1), GetInputShape(q, 0), m});
        // reshape
        LOOP("L1", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)){
            (void)index;
            ReshapeInplace(q, q_reshape);
        }
        // view + op
        LOOP("L2", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange(GetInputShape(q_reshape, 0))){
            Tensor tmp0 = View(q_reshape, {1, GetInputShape(q_reshape, 1), m}, {loopIdx, 0, 0});
            TileShape::Current().SetVecTile(16, 16, 16);
            auto tmp = AddS(tmp0, Element(tmp0->Datatype(), 0.01));
            Assemble(tmp, {loopIdx, 0, 0}, out);
        }
    }
    d = 24;
    Tensor q_real(DT_FP32, {sq, d, m});
    Tensor out_real(DT_FP32, {d, sq, m});
    genDateAndExe(q_real, out_real, 1);
}

TEST_F(DynamicReshapeUnalignImplaceTest, test_reshape_special) {
    TileShape::Current().SetVecTile(1, 16, 16);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int sq = 8;
    int d = -1;
    int m = 8;
    std::vector<int64_t> qShape3Dim = {sq, d, m}; // (8, -1(2), 8)
    std::vector<int64_t> qShape2Dim = {4, 4, m}; // (4, -1(4), 8) 

    Tensor q(DT_FP32, qShape3Dim, "q");
    Tensor out(DT_FP32, qShape2Dim, "out");

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {q}, {out}) {
        Tensor q_reshape(DT_FP32, {GetInputShape(q, 0) * GetInputShape(q, 1) / 4, 4, m});
        // reshape
        LOOP("L1", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)){
            (void)index;
            ReshapeInplace(q, q_reshape);
        }
        // view + op
        LOOP("L2", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange(GetInputShape(q_reshape, 0))){
            Tensor tmp0 = View(q_reshape, {1, GetInputShape(q_reshape, 1), m}, {loopIdx, 0, 0});
            auto tmp = AddS(tmp0, Element(tmp0->Datatype(), 0.01));
            Assemble(tmp, {loopIdx, 0, 0}, out);
        }
    }
    d = 2;
    Tensor q_real(DT_FP32, {sq, d, m});
    Tensor out_real(DT_FP32, {4, 4, m});
    genDateAndExe(q_real, out_real, 1);
}

TEST_F(DynamicReshapeUnalignImplaceTest, test_op_reshape_op) {
    TileShape::Current().SetVecTile(1, 4, 32);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetPassConfig("PVC2_OOO", "SplitReshape", "DISABLE_PASS", true);

    int b = 2;
    int sq = -1;
    int d = 8;
    int bSq = (sq == -1) ? -1 : b*sq;
    std::vector<int64_t> qShape3Dim = {b, sq, d}; //{2, -1(18), 8}
    std::vector<int64_t> qShape2Dim = {bSq, d};  //{-1(36), 8}
    Tensor q(DT_FP32, qShape3Dim, "q");
    Tensor out(DT_FP32, qShape2Dim, "out");

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {q}, {out}) {   
        // op
        Tensor addTmp(DT_FP32, {b, GetInputShape(q, 1), d});
        LOOP("L1", FunctionType::DYNAMIC_LOOP, sqIdx, LoopRange(GetInputShape(q, 1))) {
            Tensor viewTmp = View(q, {b, 1, d}, {0, sqIdx, 0});
            auto res = AddS(viewTmp, Element(viewTmp->Datatype(), 0.01));
            Assemble(res, {0, sqIdx, 0}, addTmp);
        }
        Tensor qReshape(DT_FP32, {GetInputShape(addTmp, 0) * GetInputShape(addTmp, 1) , d});
        // reshape
        LOOP("ReshapeInplace", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)){
            (void)idx;
            ReshapeInplace(addTmp, qReshape);
        }
        // view + op
        int offSet = 32;
        LOOP("L2", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange((GetInputShape(addTmp, 0) * GetInputShape(addTmp, 1) + offSet - 1)/offSet)){
            TileShape::Current().SetVecTile(4, 32);
            Tensor tmp0 = View(qReshape, {offSet, d}, {min(GetInputShape(addTmp, 0) * GetInputShape(addTmp, 1) - loopIdx * offSet, offSet), d}, {loopIdx * offSet, 0});
            Tensor tmp = AddS(tmp0, Element(tmp0->Datatype(), 0.01));
            Assemble(tmp, {loopIdx * offSet, 0}, out);
        }
    }
    sq = 18;
    Tensor q_real(DT_FP32, {b, sq, d});
    Tensor out_real(DT_FP32, {b * sq, d});
    genDateAndExe(q_real, out_real, 2);
}

TEST_F(DynamicReshapeUnalignImplaceTest, test_src_op_dst_op) {
    TileShape::Current().SetVecTile(1, 16, 16);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    int sq = 5;
    int d = -1;
    int m = 8;
    int sqD = (d == -1) ? -1 : sq*d;
    std::vector<int64_t> qShape3Dim = {sq, d, m}; //(5, -1(5), 8)
    std::vector<int64_t> qShape2Dim = {sqD, m};   //(-1(5*5), 8)

    Tensor q(DT_FP32, qShape3Dim, "q");
    Tensor outSrc(DT_FP32, qShape3Dim, "outSrc");
    Tensor outDst(DT_FP32, qShape2Dim, "outDst");
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {q}, {outSrc, outDst}) {
        Tensor q_reshape(DT_FP32, {GetInputShape(q, 0) * GetInputShape(q, 1), m});
        LOOP("reshapeInplace", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)){
            (void)index;
            ReshapeInplace(q, q_reshape);
        }
        LOOP("srcOp", FunctionType::DYNAMIC_LOOP, indx, LoopRange(GetInputShape(q, 1))){
            Tensor tmp0 = View(q, {sq, 1, m}, {0, indx, 0});
            auto tmp1 = AddS(tmp0, Element(tmp0->Datatype(), 0.02));
            Assemble(tmp1, {0, indx, 0}, outSrc);
        }
        SymbolicScalar offSet = 32;
        LOOP("destOp", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange((GetInputShape(q, 0) * GetInputShape(q, 1) + offSet - 1)/offSet)){
            Tensor tmp2 = View(q_reshape, {offSet, m}, {min(GetInputShape(q, 0) * GetInputShape(q, 1) - loopIdx * offSet, offSet), m}, {loopIdx * offSet, 0});
            TileShape::Current().SetVecTile(16, 16);
            auto tmp3 = AddS(tmp2, Element(tmp2->Datatype(), 0.01));
            Assemble(tmp3, {loopIdx * offSet, 0}, outDst);
        }
    }
    d = 5;
    Tensor qReal(DT_FP32, {sq, d, m});
    Tensor outSrcReal(DT_FP32, {sq, d, m});
    Tensor outDstReal(DT_FP32, {sq*d, m});

    std::vector<int64_t> shape = qReal.GetShape();
    size_t elementSum = 1;
    for (size_t i = 0; i < shape.size(); i++){
        elementSum *= shape[i];
    }
    std::vector<float> inputValueData(elementSum, 0);
    for (size_t i = 0; i < elementSum ; i++){
        inputValueData[i] = static_cast<float>(i);
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(qReal, inputValueData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(outSrcReal, 0.0f),
        RawTensorData::CreateConstantTensor<float>(outDstReal, 0.0f),
    });

    std::vector<float> outSrcGolden(elementSum, 1.02f);
    for (size_t i = 0; i < elementSum ; i++){
        outSrcGolden[i] = static_cast<float>(i) + 0.02f;
    }
    std::vector<float> outDstGolden(elementSum, 1.01f);
    for (size_t i = 0; i < elementSum ; i++){
        outDstGolden[i] = static_cast<float>(i) + 0.01f;
    }
    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), DeviceLauncherConfig(qReal->GetDataSize()));
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outSrcGolden, (float *)outputResult->data(), 0.001f));
    auto outsumResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    EXPECT_TRUE(resultCmp(outDstGolden, (float *)outsumResult->data(), 0.001f));
}