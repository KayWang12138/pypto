/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_onehot.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
class TestCodegenDynOneHot : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }


    void TearDown() override {}
};
TEST_F(TestCodegenDynOneHot, TestDynOpOneHot) {
    std::vector<int64_t> indicesShape = {4, 8};          // 2D input (1~3D allowed)
    int64_t depth = 16;
    std::vector<int64_t> outputShape = {4, 8, 16};       // 3D output (2~4D)

    auto shapeImme = OpImmediate::Specified(outputShape);
    TileShape::Current().SetVecTile(outputShape);

    Tensor inputStub(DT_INT32, indicesShape, "Indices");
    Tensor outputStub(DT_INT64, outputShape, "OneHotOut");

    std::string funcName = "TestDynOpOneHot";
    FUNCTION(funcName, {inputStub, outputStub}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            outputStub = Cast(inputStub, DT_INT64);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(
        FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    
    auto localIndices = CreateLogicalTensor({
        *function, DataType::DT_INT32, MemoryType::MEM_UB, indicesShape});
    auto localOnValue = CreateLogicalTensor({
        *function, DataType::DT_INT64, MemoryType::MEM_UB, std::vector<int64_t>{1}});
    auto localOffValue = CreateLogicalTensor({
        *function, DataType::DT_INT64, MemoryType::MEM_UB, std::vector<int64_t>{1}});
    auto localOutput = CreateLogicalTensor({
        *function, DataType::DT_INT64, MemoryType::MEM_UB, outputShape});

    std::vector<SymbolicScalar> dynIndicesShape = {4, 8};
    std::vector<SymbolicScalar> dynOutputShape = {4, 8, 16};
    localIndices->UpdateDynValidShape(dynIndicesShape);
    localOnValue->UpdateDynValidShape({1});
    localOffValue->UpdateDynValidShape({1});
    localOutput->UpdateDynValidShape(dynOutputShape);

    auto &op = function->AddOperation(
        Opcode::OP_ONE_HOT,
        {localIndices, localOnValue, localOffValue},
        {localOutput});
    op.SetAttribute("depth", static_cast<int64_t>(depth));
    op.SetAttribute("axis", static_cast<int64_t>(-1));
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);

    function->GetTensorMap().inverseMap_[localIndices->GetMagic()] = localIndices;
    function->GetTensorMap().inverseMap_[localOnValue->GetMagic()] = localOnValue;
    function->GetTensorMap().inverseMap_[localOffValue->GetMagic()] = localOffValue;
    function->GetTensorMap().inverseMap_[localOutput->GetMagic()] = localOutput;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynOneHot<int32_t, int64_t, /*IndicesRank*/ 2, 4, 8, /*Depth*/ 16, /*Axis*/ -1>((__ubuf__ int32_t*)UB_S0_E0, (__ubuf__ int64_t*)UB_S0_E1, (__ubuf__ int64_t*)UB_S0_E2, (__ubuf__ int64_t*)UB_S0_E3);
)!!!";
    EXPECT_EQ(res, expect);
}
}