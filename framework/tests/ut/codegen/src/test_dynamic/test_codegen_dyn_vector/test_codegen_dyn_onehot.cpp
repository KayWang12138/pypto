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
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};
void TestOneHotBody(int dim = 2) {
    std::vector<int64_t> indicesShape = {8, 16};
    std::vector<int64_t> outputShape = {8, 16, 64}; // depth=64
    std::vector<SymbolicScalar> dynValidShape = {8, 16};
    std::vector<SymbolicScalar> dynOutputShape = {8, 16, 64};
    if (dim == 3) {
        indicesShape = {8, 16, 32};
        outputShape = {8, 16, 32, 64}; // depth=64
        dynValidShape = {8, 16, 32};
        dynOutputShape = {8, 16, 32, 64};
    }
    auto shapeImme = OpImmediate::Specified(outputShape);
    TileShape::Current().SetVecTile(outputShape);

    Tensor inputStub(DT_INT32, indicesShape, "indices");
    Tensor outputStub(DT_INT64, outputShape, "onehot");

    std::string funcName = "ONEHOT";
    FUNCTION(funcName, {inputStub, outputStub}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            outputStub = Add(inputStub, inputStub);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    auto localIndices =
        CreateLogicalTensor({*function, DataType::DT_INT32, MemoryType::MEM_UB, indicesShape, dynValidShape});
    auto localOutput = CreateLogicalTensor({*function, DataType::DT_INT64, MemoryType::MEM_UB, outputShape, dynOutputShape});

    auto &op = function->AddOperation(Opcode::OP_ONE_HOT, {localIndices}, {localOutput});
    op.SetAttribute(OP_ATTR_PREFIX + "shape", outputShape);
    op.SetAttribute("depth", static_cast<int64_t>(64));
    op.SetAttribute("axis", static_cast<int64_t>(-1));
    op.SetOOpAttrOffset(0, 0);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localIndices->GetMagic()] = localIndices;
    function->GetTensorMap().inverseMap_[localOutput->GetMagic()] = localOutput;

    cop.Init(op);
    cop.GenOpCode();
}

TEST_F(TestCodegenDynOneHot, OneHotDim2) {
    TestOneHotBody();
}

TEST_F(TestCodegenDynOneHot, OneHotDim3) {
    TestOneHotBody(3);
}
} // namespace npu::tile_fwk