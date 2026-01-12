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
 * \file test_matmul_operation.cpp
 * \brief
 */

#include "test_operation.h"
#include "interface/operation/operation_impl.h"

using namespace tile_fwk::test_operation;
namespace {
constexpr int scaleAIndex = 2;
constexpr int scaleBIndex = 3;
constexpr int scaleIndex = 4;
constexpr int biasIndex = 5;
constexpr int l0cToL1Index = 6;

struct MXMatmulOpFuncArgs : public OpFuncArgs {
    MXMatmulOpFuncArgs(const std::vector<int64_t> &viewShape, const std::vector<std::vector<int64_t>> &tileShape,
        const MatmulTestCaseParam &param)
        : viewShape_(viewShape), tileShape_(tileShape), param_(param) {}

    std::vector<int64_t> viewShape_;
    std::vector<std::vector<int64_t>> tileShape_;
    MatmulTestCaseParam param_;
};

struct MXMatmulOpMetaData {
    explicit MXMatmulOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static Tensor CallMatmulOp(const Tensor &tensorA, const Tensor &scaleA, const Tensor &tensorB, const Tensor &scaleB,
                           const MatmulTestCaseParam &param, const Matrix::MatmulExtendParam &matmulExtendParam)
{
    return Matrix::MatmulMX(param.outDtype, tensorA, scaleA, tensorB, scaleB, matmulExtendParam, param.transA,
                          param.transA, param.transB, param.transB, param.isCMatrixNz);
}

static void MatmulOperationExeFuncNoSplit(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    auto args = static_cast<const MXMatmulOpFuncArgs *>(opArgs);
    bool transA = args->param_.transA;
    bool transB = args->param_.transB;
    float scaleValue = args->param_.scaleValue;
    int reluTypeInt = args->param_.reluTypeInt;
    Matrix::ReLuType reluType = static_cast<Matrix::ReLuType>(reluTypeInt);
    SymbolicScalar mDim = transA ? inputs[0].GetShape()[1] : inputs[0].GetShape()[0];
    SymbolicScalar kDim = transA ? inputs[0].GetShape()[0] : inputs[0].GetShape()[1];
    SymbolicScalar nDim = transB ? inputs[1].GetShape()[0] : inputs[1].GetShape()[1];
    SymbolicScalar kScaleDim = kDim / 32;
    FUNCTION("testNoSplit",
             {inputs[0], inputs[1], inputs[scaleAIndex], inputs[scaleBIndex], inputs[scaleIndex], inputs[biasIndex]},
             {outputs[0]})
    {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1))
        {
            Tensor tensorA;
            Tensor scaleA;
            if (transA) {
                tensorA = View(inputs[0], {kDim, mDim}, {kDim, mDim}, {0, 0});
                scaleA = View(inputs[scaleAIndex], {kScaleDim, mDim}, {kScaleDim, mDim}, {0, 0});
            } else {
                tensorA = View(inputs[0], {mDim, kDim}, {mDim, kDim}, {mIdx, 0});
                scaleA = View(inputs[scaleAIndex], {mDim, kScaleDim}, {mDim, kScaleDim}, {mIdx, 0});
            }
            Tensor tensorB;
            Tensor scaleB;
            if (transB) {
                tensorB = View(inputs[1], {nDim, kDim}, {nDim, kDim}, {0, 0});
                scaleB = View(inputs[scaleBIndex], {nDim, kScaleDim}, {nDim, kScaleDim}, {0, 0});
            } else {
                tensorB = View(inputs[1], {kDim, nDim}, {kDim, nDim}, {0, 0});
                scaleB = View(inputs[scaleBIndex], {kScaleDim, nDim}, {kScaleDim, nDim}, {0, 0});
            }
            Matrix::MatmulExtendParam param;
            param.scaleValue = scaleValue;
            param.reluType = reluType;
            if (args->param_.hasScale) {
                param.scaleTensor = View(inputs[scaleIndex], {1, nDim}, {1, nDim}, {0, 0});
            }
            if (args->param_.hasBias) {
                param.biasTensor = View(inputs[biasIndex], {1, nDim}, {1, nDim}, {0, 0});
            }
            outputs[0] = CallMatmulOp(tensorA, scaleA, tensorB, scaleB, args->param_, param);
        }
    }
}

static void MXMatmulOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);

    auto args = static_cast<const MXMatmulOpFuncArgs *>(opArgs);
    if (args->param_.hasScale || args->param_.hasBias) {
        int64_t nTile =
            (args->tileShape_[2][0] < args->tileShape_[2][1]) ? args->tileShape_[2][0] : args->tileShape_[2][1];
        TileShape::Current().SetCubeTile({args->tileShape_[0][0], args->tileShape_[0][1]},
                                         {args->tileShape_[1][0], args->tileShape_[1][1]},
                                         {nTile, nTile}, true, args->param_.enableKSplit);
    } else {
        TileShape::Current().SetCubeTile({args->tileShape_[0][0], args->tileShape_[0][1]},
                                         {args->tileShape_[1][0], args->tileShape_[1][1]},
                                         {args->tileShape_[2][0], args->tileShape_[2][1]}, true, args->param_.enableKSplit);
    }

    const size_t MM_VIEW_SHAPE_DIM = 2;
    ASSERT(args->viewShape_.size() == MM_VIEW_SHAPE_DIM);
    const int64_t mView = args->viewShape_[0];
    const int64_t nView = args->viewShape_[1];
    if (mView <= 0 && nView <= 0) {
        return MatmulOperationExeFuncNoSplit(inputs, outputs, opArgs);
    }
}

static void CheckBTransNZUnaligned(bool transB, bool isCMatrixNz, const Tensor &b, const Tensor &c) {
    constexpr int blockAlignBytes = 32;
    ASSERT(BytesOf(c.GetDataType()) != 0)
        << "wrong data type";
    int innerNum = blockAlignBytes / BytesOf(c.GetDataType());
    SymbolicScalar bN = transB ? b.GetShape()[0] : b.GetShape()[1];
    SymbolicScalar cN = c.GetShape()[1];
    bool nNotEqualCase = !transB && isCMatrixNz && cN % innerNum == 0;
    // (N, K)输入，NZ输出是，由于ND2NZ指令无法在N轴补零，最终NPU输出在N轴可能存在脏数据，暂不支持。
    ASSERT(bN == cN || nNotEqualCase)
        << "N, K shape for NZ output format with N unaligned is not supported";
}

class MXMatmulOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<MXMatmulOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestMXMatmul, MXMatmulOperationTest,
    ::testing::ValuesIn(GetOpMetaData<MXMatmulOpMetaData>({MXMatmulOperationExeFunc}, "MXMatmul")));

TEST_P(MXMatmulOperationTest, TestMXMatmul) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetMXMatmulTensors(test_data, "input_tensors");
    testCase.outputTensors = GetMatmulTensors(test_data, "output_tensors");
    auto args = MXMatmulOpFuncArgs(GetViewShape(test_data), GetMatmulTileShape(test_data), GetMatmulParam(test_data));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0].GetStorage()->Symbol() + ".bin",
        GetGoldenDir() + "/" + testCase.inputTensors[1].GetStorage()->Symbol() + ".bin",
        GetGoldenDir() + "/scale0.bin",
        GetGoldenDir() + "/scale1.bin"};
    // scale tensor
    Tensor scaleTensor = GetParamTensor(test_data, "scale_tensors");
    testCase.inputTensors.push_back(scaleTensor);
    if (scaleTensor.GetStorage() != nullptr) {
        testCase.inputPaths.push_back(GetGoldenDir() + "/" + scaleTensor.GetStorage()->Symbol() + ".bin");
    } else {
        testCase.inputPaths.push_back("");
    }
    // bias tensor
    Tensor biasTensor = GetParamTensor(test_data, "bias_tensors");
    testCase.inputTensors.push_back(biasTensor);
    if (biasTensor.GetStorage() != nullptr) {
        testCase.inputPaths.push_back(GetGoldenDir() + "/" + biasTensor.GetStorage()->Symbol() + ".bin");
    } else {
        testCase.inputPaths.push_back("");
    }
    if (args.param_.enable_l0c2l1) {
        Tensor l0c2L1Tensor = GetParamTensor(test_data, "l0c2l1_tensor");
        testCase.inputTensors.push_back(l0c2L1Tensor);
        testCase.inputPaths.push_back(GetGoldenDir() + "/" + l0c2L1Tensor.GetStorage()->Symbol() + ".bin");
    }
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    CheckBTransNZUnaligned(
        args.param_.transB, args.param_.isCMatrixNz, testCase.inputTensors[1], testCase.outputTensors[0]);
    if (args.param_.enableKSplit) {
        TestExecutor::setGMNotClear();
    }
    TestExecutor::runTest(testCase);
}
} // namespace