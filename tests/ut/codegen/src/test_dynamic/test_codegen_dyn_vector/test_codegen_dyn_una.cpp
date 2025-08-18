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
 * \file test_codegen_dyn_una.cpp
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
class TestCodegenDynUna : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynUna, TestAbsDynamic) {
    int S0 = 8;
    int S1 = 4608;
    int D0 = 8;
    int D1 = 4608;

    std::vector<int> srcShape = {S0, S1};
    std::vector<int> dstShape = {D0, D1};

    Program::GetInstance().GetTileShape().SetVecTileShapes({8, 128});
    Tensor input_a(DataType::DT_FP16, srcShape, "A");
    Tensor output(DataType::DT_FP16, dstShape, "C");

    FUNCTION("ABS_T", FunctionType::STATIC, {input_a, output}) {
        output = Abs(input_a);
    }
    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_ABS_T");
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    ConfigManager::Instance().SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    for (auto &subFunc : function->rootFunc_->programs_) {
        for (auto &op : subFunc.second->Operations()) {
            if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode()) || OpcodeManager::Inst().IsCopyOut(op.GetOpcode())) {
                if (IsCopyIn(op.GetOpcode()))
                    op.SetIOpAttrOffset(0, 0);
                else
                    op.SetOOpAttrOffset(0, 0);
                op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
            }
        }
        DynParamInfo fakeParam = {3, 0, 0, DynParamInfoType::VALID_SHAPE, 0};
        subFunc.second->dynParamTable_.emplace("sym_113_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_113_dim_1", fakeParam);
    }

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynUna, TestDynBitSort) {
    std::vector<int> shape = {64, 64};

    auto shapeImme = OpImmediate::Specified(shape);
    Program::GetInstance().GetTileShape().SetVecTileShapes(shape);
    ConfigManager::Instance().SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, FunctionType::STATIC, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    localTensor->UpdateSubgraphID(0);
    localTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    localTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    localTensor->SetMagic(OP_MAGIC3);
    localTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localTensor->UpdateDynValidShape(dynValidShape);
    localTensor->memorymap[0].memId = 0;
    localTensor->memorymap[0].start = 0;
    localTensor->memorymap[0].end = 0;

    auto localOutTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    localOutTensor->UpdateSubgraphID(0);
    localOutTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    localOutTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    localOutTensor->SetMagic(OP_MAGIC4);
    localOutTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localOutTensor->UpdateDynValidShape(dynValidShape);
    localOutTensor->memorymap[0].memId = 1;
    localOutTensor->memorymap[0].start = 1;
    localOutTensor->memorymap[0].end = 1;

    auto &op = function->AddOperation(Opcode::OP_BITSORT, {localTensor}, {localOutTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", 1);
    op.SetAttribute(OP_ATTR_PREFIX + "order", 1);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, memAlloc);
    CodeGenOpCloudNPU cop(memAlloc, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynBitSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 1, 1>((__ubuf__ float*)UB_S1_E1, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynUna, TestDynMrgSort) {
    std::vector<int> shape = {64, 64};

    auto shapeImme = OpImmediate::Specified(shape);
    Program::GetInstance().GetTileShape().SetVecTileShapes(shape);
    ConfigManager::Instance().SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, FunctionType::STATIC, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    localTensor->UpdateSubgraphID(0);
    localTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    localTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    localTensor->SetMagic(OP_MAGIC3);
    localTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localTensor->UpdateDynValidShape(dynValidShape);
    localTensor->memorymap[0].memId = 0;
    localTensor->memorymap[0].start = 0;
    localTensor->memorymap[0].end = 0;

    auto localOutTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    localOutTensor->UpdateSubgraphID(0);
    localOutTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    localOutTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    localOutTensor->SetMagic(OP_MAGIC4);
    localOutTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localOutTensor->UpdateDynValidShape(dynValidShape);
    localOutTensor->memorymap[0].memId = 1;
    localOutTensor->memorymap[0].start = 1;
    localOutTensor->memorymap[0].end = 1;

    auto &op = function->AddOperation(Opcode::OP_MRGSORT, {localTensor}, {localOutTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", 1);
    op.SetAttribute(OP_ATTR_PREFIX + "order", 1);
    op.SetAttribute(OP_ATTR_PREFIX + "kvalue", 1);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, memAlloc);
    CodeGenOpCloudNPU cop(memAlloc, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;
    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynMrgSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 1, 1, 1>((__ubuf__ float*)UB_S1_E1, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynUna, TestDynExtract) {
    std::vector<int> shape = {64, 64};

    auto shapeImme = OpImmediate::Specified(shape);
    Program::GetInstance().GetTileShape().SetVecTileShapes(shape);
    ConfigManager::Instance().SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, FunctionType::STATIC, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    localTensor->UpdateSubgraphID(0);
    localTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    localTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    localTensor->SetMagic(OP_MAGIC3);
    localTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localTensor->UpdateDynValidShape(dynValidShape);
    localTensor->memorymap[0].memId = 0;
    localTensor->memorymap[0].start = 0;
    localTensor->memorymap[0].end = 0;

    auto localOutTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    localOutTensor->UpdateSubgraphID(0);
    localOutTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    localOutTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    localOutTensor->SetMagic(OP_MAGIC4);
    localOutTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localOutTensor->UpdateDynValidShape(dynValidShape);
    localOutTensor->memorymap[0].memId = 1;
    localOutTensor->memorymap[0].start = 1;
    localOutTensor->memorymap[0].end = 1;

    auto &op = function->AddOperation(Opcode::OP_EXTRACT, {localTensor}, {localOutTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "kvalue", 1);
    op.SetAttribute(OP_ATTR_PREFIX + "mode", 1);
    op.SetAttribute(OP_ATTR_PREFIX + "order", 1);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, memAlloc);
    CodeGenOpCloudNPU cop(memAlloc, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;
    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynExtract<float, float, 1, 64, 64, 1, 1, 1>((__ubuf__ float*)UB_S1_E1, (__ubuf__ float*)UB_S0_E0, 1, 1, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynUna, TestDynExpand) {
    std::vector<int> shape = {64, 64};
    std::vector<int> shape1 = {1, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    Program::GetInstance().GetTileShape().SetVecTileShapes(shape);
    ConfigManager::Instance().SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, FunctionType::STATIC, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape1 = {1, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape1});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    localTensor->UpdateDynValidShape(dynValidShape1);
    localOutTensor->UpdateDynValidShape(dynValidShape);

    auto &op = function->AddOperation(Opcode::OP_EXPAND, {localTensor}, {localOutTensor});
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", 0);

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, memAlloc);
    CodeGenOpCloudNPU cop(memAlloc, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect = R"!!!(TileOp::DynTexpand_<float, /*DS*/ 1, 64, 64, /*SS*/ 1, 1, 64, 2>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64, 1, 1, 1, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk