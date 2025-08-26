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
 * \file test_matmul_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct MatmulOpFuncArgs : public OpFuncArgs {
    MatmulOpFuncArgs(const std::vector<int> &viewShape, const std::vector<std::vector<int>> &tileShape,
        const MatmulTestCaseParam &param)
        : viewShape_(viewShape), tileShape_(tileShape), param_(param) {}

    std::vector<int> viewShape_;
    std::vector<std::vector<int>> tileShape_;
    MatmulTestCaseParam param_;
};

struct MatmulOpMetaData {
    explicit MatmulOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static Tensor CallMatmulOp(const Tensor &tensorA, const Tensor &tensorB, const MatmulTestCaseParam &param) {
    if (!param.transA && !param.transB && !param.isCMatrixNz) {
        return Matrix::Matmul<false, false, false>(param.outDtype, tensorA, tensorB);
    } else if (!param.transA && !param.transB && param.isCMatrixNz) {
        return Matrix::Matmul<false, false, true>(param.outDtype, tensorA, tensorB);
    } else if (!param.transA && param.transB && !param.isCMatrixNz) {
        return Matrix::Matmul<false, true, false>(param.outDtype, tensorA, tensorB);
    } else if (!param.transA && param.transB && param.isCMatrixNz) {
        return Matrix::Matmul<false, true, true>(param.outDtype, tensorA, tensorB);
    } else if (param.transA && !param.transB && !param.isCMatrixNz) {
        return Matrix::Matmul<true, false, false>(param.outDtype, tensorA, tensorB);
    } else if (param.transA && !param.transB && param.isCMatrixNz) {
        return Matrix::Matmul<true, false, true>(param.outDtype, tensorA, tensorB);
    } else if (param.transA && param.transB && !param.isCMatrixNz) {
        return Matrix::Matmul<true, true, false>(param.outDtype, tensorA, tensorB);
    } else {
        return Matrix::Matmul<true, true, true>(param.outDtype, tensorA, tensorB);
    }
}

static void MatmulOperationExeFuncNoSplit(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);

    auto args = static_cast<const MatmulOpFuncArgs *>(opArgs);
    bool transA = args->param_.transA;
    bool transB = args->param_.transB;
    SymbolicScalar mDim = transA ? inputs[0]->shape[1] : inputs[0]->shape[0];
    SymbolicScalar kDim = transA ? inputs[0]->shape[0] : inputs[0]->shape[1];
    SymbolicScalar nDim = transB ? inputs[1]->shape[0] : inputs[1]->shape[1];

    FUNCTION("testNoSplit", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1)) {
            Tensor tensorA;
            if (transA) {
                tensorA = DViewPad(inputs[0], {kDim, mDim}, {kDim, mDim}, {0, 0});
            } else {
                tensorA = DViewPad(inputs[0], {mDim, kDim}, {mDim, kDim}, {mIdx, 0});
            }
            Tensor tensorB;
            if (transB) {
                tensorB = DViewPad(inputs[1], {nDim, kDim}, {nDim, kDim}, {0, 0});
            } else {
                tensorB = DViewPad(inputs[1], {kDim, nDim}, {kDim, nDim}, {0, 0});
            }

            Program::GetInstance().GetTileShape().SetCubeTileShapes({args->tileShape_[0][0], args->tileShape_[0][1]},
                {args->tileShape_[1][0], args->tileShape_[1][1]}, {args->tileShape_[2][0], args->tileShape_[2][1]});
            outputs[0] = CallMatmulOp(tensorA, tensorB, args->param_);
        }
    }
}

static void MatmulOperationExeFuncSplitM(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);

    auto args = static_cast<const MatmulOpFuncArgs *>(opArgs);
    const int mView = args->viewShape_[0];
    bool transA = args->param_.transA;
    bool transB = args->param_.transB;
    SymbolicScalar mDim = transA ? inputs[0]->shape[1] : inputs[0]->shape[0];
    SymbolicScalar kDim = transA ? inputs[0]->shape[0] : inputs[0]->shape[1];
    SymbolicScalar nDim = transB ? inputs[1]->shape[0] : inputs[1]->shape[1];

    FUNCTION("testMSplit", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, CeilDivSymbolicScalar(mDim, mView), 1)) {
            Tensor tensorA;
            if (transA) {
                tensorA =
                    DViewPad(inputs[0], {kDim, mView}, {kDim, std::min(mDim - mView * mIdx, mView)}, {0, mIdx * mView});
            } else {
                tensorA =
                    DViewPad(inputs[0], {mView, kDim}, {std::min(mDim - mView * mIdx, mView), kDim}, {mIdx * mView, 0});
            }
            Tensor tensorB;
            if (transB) {
                tensorB = DViewPad(inputs[1], {nDim, kDim}, {nDim, kDim}, {0, 0});
            } else {
                tensorB = DViewPad(inputs[1], {kDim, nDim}, {kDim, nDim}, {0, 0});
            }

            Program::GetInstance().GetTileShape().SetCubeTileShapes({args->tileShape_[0][0], args->tileShape_[0][1]},
                {args->tileShape_[1][0], args->tileShape_[1][1]}, {args->tileShape_[2][0], args->tileShape_[2][1]});
            Tensor tensorC = CallMatmulOp(tensorA, tensorB, args->param_);
            DAssemble(tensorC, {mIdx * mView, 0}, outputs[0]);
        }
    }
}

static void MatmulOperationExeFuncSplitN(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);

    auto args = static_cast<const MatmulOpFuncArgs *>(opArgs);
    bool transA = args->param_.transA;
    bool transB = args->param_.transB;
    SymbolicScalar mDim = transA ? inputs[0]->shape[1] : inputs[0]->shape[0];
    SymbolicScalar kDim = transA ? inputs[0]->shape[0] : inputs[0]->shape[1];
    SymbolicScalar nDim = transB ? inputs[1]->shape[0] : inputs[1]->shape[1];
    const int nView = args->viewShape_[1];

    FUNCTION("testNSplit", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, CeilDivSymbolicScalar(nDim, nView), 1)) {
            Tensor tensorA;
            if (transA) {
                tensorA = DViewPad(inputs[0], {kDim, mDim}, {kDim, mDim}, {0, 0});
            } else {
                tensorA = DViewPad(inputs[0], {mDim, kDim}, {mDim, kDim}, {0, 0});
            }
            Tensor tensorB;
            if (transB) {
                tensorB =
                    DViewPad(inputs[1], {nView, kDim}, {std::min(nDim - nIdx * nView, nView), kDim}, {nIdx * nView, 0});
            } else {
                tensorB =
                    DViewPad(inputs[1], {kDim, nView}, {kDim, std::min(nDim - nIdx * nView, nView)}, {0, nIdx * nView});
            }

            Program::GetInstance().GetTileShape().SetCubeTileShapes({args->tileShape_[0][0], args->tileShape_[0][1]},
                {args->tileShape_[1][0], args->tileShape_[1][1]}, {args->tileShape_[2][0], args->tileShape_[2][1]});
            Tensor tensorC = CallMatmulOp(tensorA, tensorB, args->param_);
            DAssemble(tensorC, {0, nIdx * nView}, outputs[0]);
        }
    }
}

static void MatmulOperationExeFuncSplitMN(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);

    auto args = static_cast<const MatmulOpFuncArgs *>(opArgs);
    bool transA = args->param_.transA;
    bool transB = args->param_.transB;
    SymbolicScalar mDim = transA ? inputs[0]->shape[1] : inputs[0]->shape[0];
    SymbolicScalar kDim = transA ? inputs[0]->shape[0] : inputs[0]->shape[1];
    SymbolicScalar nDim = transB ? inputs[1]->shape[0] : inputs[1]->shape[1];
    const int mView = args->viewShape_[0];
    const int nView = args->viewShape_[1];

    FUNCTION("testMNSplit", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, CeilDivSymbolicScalar(mDim, mView), 1)) {
            LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, CeilDivSymbolicScalar(nDim, nView), 1)) {
                Tensor tensorA;
                if (transA) {
                    tensorA = DViewPad(
                        inputs[0], {kDim, mView}, {kDim, std::min(mDim - mView * mIdx, mView)}, {0, mIdx * mView});
                } else {
                    tensorA = DViewPad(
                        inputs[0], {mView, kDim}, {std::min(mDim - mView * mIdx, mView), kDim}, {mIdx * mView, 0});
                }
                Tensor tensorB;
                if (transB) {
                    tensorB = DViewPad(
                        inputs[1], {nView, kDim}, {std::min(nDim - nIdx * nView, nView), kDim}, {nIdx * nView, 0});
                } else {
                    tensorB = DViewPad(
                        inputs[1], {kDim, nView}, {kDim, std::min(nDim - nIdx * nView, nView)}, {0, nIdx * nView});
                }

                Program::GetInstance().GetTileShape().SetCubeTileShapes(
                    {args->tileShape_[0][0], args->tileShape_[0][1]}, {args->tileShape_[1][0], args->tileShape_[1][1]},
                    {args->tileShape_[2][0], args->tileShape_[2][1]});
                Tensor tensorC = CallMatmulOp(tensorA, tensorB, args->param_);
                DAssemble(tensorC, {mIdx * mView, nIdx * nView}, outputs[0]);
            }
        }
    }
}

static void MatmulOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    auto args = static_cast<const MatmulOpFuncArgs *>(opArgs);
    const size_t MM_VIEW_SHAPE_DIM = 2;
    ASSERT(args->viewShape_.size() == MM_VIEW_SHAPE_DIM);
    const int mView = args->viewShape_[0];
    const int nView = args->viewShape_[1];
    if (mView > 0 && nView > 0) {
        return MatmulOperationExeFuncSplitMN(inputs, outputs, opArgs);
    } else if (mView > 0) {
        return MatmulOperationExeFuncSplitM(inputs, outputs, opArgs);
    } else if (nView > 0) {
        return MatmulOperationExeFuncSplitN(inputs, outputs, opArgs);
    } else {
        return MatmulOperationExeFuncNoSplit(inputs, outputs, opArgs);
    }
}

class MatmulOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<MatmulOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestMatmul, MatmulOperationTest,
    ::testing::ValuesIn(GetOpMetaData<MatmulOpMetaData>({MatmulOperationExeFunc}, "Matmul")));

TEST_P(MatmulOperationTest, TestMatmul) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetMatmulTensors(test_data, "input_tensors");
    testCase.outputTensors = GetMatmulTensors(test_data, "output_tensors");
    auto args = MatmulOpFuncArgs(GetViewShape(test_data), GetMatmulTileShape(test_data), GetMatmulParam(test_data));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0]->Symbol() + ".bin",
        GetGoldenDir() + "/" + testCase.inputTensors[1]->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0]->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}
} // namespace