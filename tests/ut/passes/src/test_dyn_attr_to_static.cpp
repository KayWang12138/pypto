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
 * \file test_dyn_attr_to_static.cpp
 * \brief Unit test for DynAttrToStatic pass.
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/block_graph_pass/dyn_attr_to_static.h"
#include "interface/interpreter/raw_tensor_data.h"

using namespace npu::tile_fwk;

class DynAttrToStaticTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        config::SetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, true);
        config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", "PRINT_FUNCTION", true);
    }
    void TearDown() override {}
};

const std::string LEFT_BRACKER = "(";
const std::string MAYBE_CONST_POSTFIX = "MAYBE_CONST";

bool VerifyNewMacroExpr(std::reference_wrapper<SymbolicScalar>& dynScalar) {
    std::string dynParamExpr = SymbolicExpressionTable::BuildExpression(dynScalar);

    // COA类型宏格式是"(RUNTIME_GET_COA_XXX("
    if (dynParamExpr.find(COA_PREFIX) != 1) {// 只保留COA类型宏，进行下一步检查
        return true;
    }

    // 找到第二个"("的位置
    size_t secondLBracket = dynParamExpr.find(LEFT_BRACKER, 1);

    // 检查找到第二个"("前的字符是否是"MAYBE_CONST"
    size_t postfixLen = MAYBE_CONST_POSTFIX.length();
    if (secondLBracket <= postfixLen) {
        return false;
    }
    if (dynParamExpr.substr(secondLBracket - postfixLen, postfixLen) == MAYBE_CONST_POSTFIX) {
        return true;
    }
    return false;
}

TEST_F(DynAttrToStaticTest, TestGetTensorData) {
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = n * 8;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }

    Tensor inputC(DT_FP32, {n, s}, "inputC");
    std::vector<float> inputCData(n * s);
    for (int k = 0; k < n * s; k++) {
        inputCData[k] = (float)(1.0 * ((k % s) / n));
    }
    Tensor output(DT_FP32, {n, n}, "output");
    std::vector<float> outputGolden(n * n, 12.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputC, inputCData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    FunctionConfig funConfig;
    FUNCTION("test_coa", funConfig, {inputA, inputC}, {output}) {
        LOOP("loop", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Tensor t0 = AddS(inputA, Element(DT_INT32, (int64_t)2)); // t0[i, j] -> inputA[i, j] + 2 -> i * n + j + 2
            SymbolicScalar v0 = GetTensorData(t0, {0, 1}); // t0[0, 1] -> 0 * n + 1 + 2 -> 3
            SymbolicScalar v1 = GetTensorData(t0, {0, 2}); // t0[0, 2] -> 0 * n + 2 + 2 -> 4
            auto t2 = View(inputC, {n, n}, {0, v0 * n});
            auto t3 = View(inputC, {n, n}, {0, v1 * n});
            output = Mul(t2, t3);
        }
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_test_coa");
    npu::tile_fwk::DynAttrToStatic passDynAttrToStatic;
    passDynAttrToStatic.RunOnFunction(*func);

    // ================== Verify Pass Effect TENSOR_loop ==================
    std::string loopPathFuncName = "TENSOR_loop_Unroll1_PATH0";
    Function* loopPathFunc = Program::GetInstance().GetFunctionByRawName(loopPathFuncName);
    Function* rootFunc = loopPathFunc->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    for (auto it = rootFunc->programs_.begin(); it != rootFunc->programs_.end(); it++) {
        Function* leafFunc = it->second;
        auto operationViewer = leafFunc->Operations(false);
        for (size_t j = 0; j < operationViewer.size(); j++) {
            auto &op = operationViewer[j];
            std::vector<std::reference_wrapper<SymbolicScalar>> dynScalarList = passDynAttrToStatic.GetOpDynamicAttributeList(op);
            for (auto dynScalar : dynScalarList) {
                EXPECT_EQ(VerifyNewMacroExpr(dynScalar), true);
            }
        }
    }
}

TEST_F(DynAttrToStaticTest, TestSetTensorData) {
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    Tensor output(DT_INT32, {n, n, n}, "output");
    std::vector<int32_t> outputGolden(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        outputGolden[i] = i;
    }

    ProgramData::GetInstance().AppendInputs({
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FunctionConfig funConfig;
    FUNCTION("test_coa", funConfig, {}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(n)) {
                for (int k = 0; k < n; k++) {
                    SetTensorData(i * tiling * tiling + j * tiling + k, {i, j, k}, output);
                }
            }
        }
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_test_coa");
    npu::tile_fwk::DynAttrToStatic passDynAttrToStatic;
    passDynAttrToStatic.RunOnFunction(*func);

    // ================== Verify Pass Effect TENSOR_Step1==================
    std::string loopPathFuncName = "TENSOR_Step1_Unroll1_PATH0";
    Function* loopPathFunc = Program::GetInstance().GetFunctionByRawName(loopPathFuncName);
    Function* rootFunc = loopPathFunc->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    for (auto it = rootFunc->programs_.begin(); it != rootFunc->programs_.end(); it++) {
        Function* leafFunc = it->second;
        auto operationViewer = leafFunc->Operations(false);
        for (size_t j = 0; j < operationViewer.size(); j++) {
            auto &op = operationViewer[j];
            std::vector<std::reference_wrapper<SymbolicScalar>> dynScalarList = passDynAttrToStatic.GetOpDynamicAttributeList(op);
            for (auto dynScalar : dynScalarList) {
                EXPECT_EQ(VerifyNewMacroExpr(dynScalar), true);
            }
        }
    }
}

TEST_F(DynAttrToStaticTest, TestDynExpression) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    TileShape::Current().SetVecTile(64, 64);

    int b = 1;
    int sq = 128;
    int d = 64;
    std::vector<int64_t> qShape = {b, d};
    std::vector<int64_t> outShape = {b * sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, outShape, "out");
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    FunctionConfig funConfig;
    FUNCTION("test_coa", funConfig, {q}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
            Tensor q0 = View(q, {1, d}, {1, d}, {batchId, 0});
            auto tmp = Expand(q0, {100, d});
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_test_coa");
    npu::tile_fwk::DynAttrToStatic passDynAttrToStatic;
    passDynAttrToStatic.RunOnFunction(*func);

    // ================== Verify Pass Effect TENSOR_L0 ==================
    std::string loopPathFuncName = "TENSOR_L0_Unroll1_PATH0";
    Function* loopPathFunc = Program::GetInstance().GetFunctionByRawName(loopPathFuncName);
    Function* rootFunc = loopPathFunc->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    for (auto it = rootFunc->programs_.begin(); it != rootFunc->programs_.end(); it++) {
        Function* leafFunc = it->second;
        for (const auto &dynParam : leafFunc->GetDynParamTable()) {
            if (dynParam.second.dim.IsValid()) {
                std::reference_wrapper<SymbolicScalar> dynExpr = const_cast<SymbolicScalar&>(dynParam.second.dim);
                EXPECT_EQ(VerifyNewMacroExpr(dynExpr), true);
            }
        }
    }
}