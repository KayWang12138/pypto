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
 * \file test_codegen_dyn_sort.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {
constexpr const unsigned OP_MAGIC3 = 3;
constexpr const unsigned OP_MAGIC4 = 4;
class TestCodegenDynSort : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

struct TestContext {
    Function *function;
    std::shared_ptr<LogicalTensor> localTensor;
    std::shared_ptr<LogicalTensor> localOutTensor;
    Operation *op;
};

std::string generateCodeForOp(Operation *op) {
    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(*op, memAlloc);
    CodeGenOpCloudNPU cop(memAlloc, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    cop.Init(*op);
    return cop.GenOpCode();
}

TestContext prepareSortParamForUT(Opcode opcode) {
    std::vector<int64_t> shape = {64, 64};

    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    ConfigManager::Instance().SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FunctionConfig funConfig(FunctionType::STATIC);
    ;
    FUNCTION(funcName, funConfig, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, OP_MAGIC3, dynValidShape});
    auto localOutTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, OP_MAGIC4, dynValidShape});

    auto &op = function->AddOperation(opcode, {localTensor}, {localOutTensor});

    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    TestContext param;
    param.function = function;
    param.localTensor = localTensor;
    param.localOutTensor = localOutTensor;
    param.op = &op;
    return param;
}

TEST_F(TestCodegenDynSort, TestDynBitSort) {
    auto param = prepareSortParamForUT(Opcode::OP_BITSORT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "axis", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::string res = generateCodeForOp(param.op);
    std::string expect =
        R"!!!(TileOp::DynBitSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynMrgSort) {
    auto param = prepareSortParamForUT(Opcode::OP_MRGSORT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "axis", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "kvalue", 1);
    param.op->SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::string res = generateCodeForOp(param.op);
    std::string expect =
        R"!!!(TileOp::DynMrgSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 1, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynExtract) {
    auto param = prepareSortParamForUT(Opcode::OP_EXTRACT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "kvalue", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "mode", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::string res = generateCodeForOp(param.op);
    std::string expect =
        R"!!!(TileOp::DynExtract<float, float, 1, 64, 64, 1, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64);
)!!!";
    EXPECT_EQ(res, expect);
}
} // namespace npu::tile_fwk