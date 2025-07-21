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
 * \file test_codegen_dyn_indexoutcast.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen_op.h"
#include "codegen/codegen_symbol.h"
#include "passes/pass_manager.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

using namespace npu::tile_fwk;

class TestCodegenDynIndexOutCast : public ::testing::Test {
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

TEST_F(TestCodegenDynIndexOutCast, IndexOutCast) {
    int S = 1;
    int S2 = 16;
    int kvLoraRank = 8;
    int qkRopeHeadDim = 8;

    std::vector<int> shape0 = {S2, kvLoraRank + qkRopeHeadDim}; // [16, 16]
    std::vector<int> shape1 = {1, S};
    std::vector<int> shape2 = {S, kvLoraRank + qkRopeHeadDim}; // [1, 16]

    Program::GetInstance().GetTileShape().SetVecTileShapes(16, 16);
    auto shapeImme = OpImmediate::Specified({16, 16});

    Tensor kv_len(DataType::DT_INT64, shape1, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, "key_states"); // [16,16]

    std::string funcName = "ScatterUpdate";
    FUNCTION(funcName, FunctionType::STATIC, {kv_len, key_states, past_key_states}) {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(DataType::DT_FP32, shape0, "IndexOutCast", 123);
    const std::vector<int> offset = {0, 0};
    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape0);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    auto localTensorSrc0 = std::make_shared<LogicalTensor>(*function, DT_FP32, shape2);
    localTensorSrc0->UpdateSubgraphID(0);
    localTensorSrc0->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    localTensorSrc0->SetMemoryTypeToBe(MemoryType::MEM_UB);
    localTensorSrc0->SetMagic(3);
    localTensorSrc0->SetAttr(OpAttributeKey::needAlloc, true);
    localTensorSrc0->memorymap[0].memId = 0;
    localTensorSrc0->memorymap[0].start = 0;
    localTensorSrc0->memorymap[0].end = 0;

    auto localTensorSrc1 = std::make_shared<LogicalTensor>(*function, DT_FP32, shape1);
    localTensorSrc1->UpdateSubgraphID(0);
    localTensorSrc1->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    localTensorSrc1->SetMemoryTypeToBe(MemoryType::MEM_UB);
    localTensorSrc1->SetMagic(3);
    localTensorSrc1->SetAttr(OpAttributeKey::needAlloc, true);
    localTensorSrc1->memorymap[0].memId = 0;
    localTensorSrc1->memorymap[0].start = 0;
    localTensorSrc1->memorymap[0].end = 0;

    auto &op =
        function->AddOperation(Opcode::OP_INDEX_OUTCAST, {localTensorSrc0, localTensorSrc1, ddrTensor}, {ddrTensor});
    op.SetAttribute("axis", 0);
    op.SetAttribute(OpAttributeKey::panzBlockSize, 1);
    std::string cacheMode = "BNSD";
    op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
    auto to_offset = OpImmediate::Specified({0, 0});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, to_offset, shapeImme, shapeImme));
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    op.SetOOpAttrOffset(0, 0);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenExtraAlloc(memAlloc, localTensorSrc0, op);
    cga.GenExtraAlloc(memAlloc, localTensorSrc1, op);
    CodeGenOpCloudNPU cop(memAlloc, function->GetTensorMap(), FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensorSrc0->GetMagic()] = localTensorSrc0;
    function->GetTensorMap().inverseMap_[localTensorSrc1->GetMagic()] = localTensorSrc1;

    cop.Init(op);

    cop.GenOpCode();
}

TEST_F(TestCodegenDynIndexOutCast, DynIndexOutUnaligned) {
    Program::GetInstance().GetTileShape().SetVecTileShapes({16, 32});

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy",
        {
            {"RemoveRedundentReshape", "RemoveRedundentReshape", PassType::TYPE_TENSOR_GRAPH},
            {        "ExpandFunction",         "ExpandFunction", PassType::TYPE_TENSOR_GRAPH},
            {         "DuplicateView",          "DuplicateView",   PassType::TYPE_TILE_GRAPH},
            {     "MergeViewAssemble",      "MergeViewAssemble",   PassType::TYPE_TILE_GRAPH},
            {      "AssignMemoryType",       "AssignMemoryType",   PassType::TYPE_TILE_GRAPH},
            {       "InsertConvertOp",        "InsertConvertOp",   PassType::TYPE_TILE_GRAPH},
            {"SplitLargeFanoutTensor", "SplitLargeFanoutTensor",   PassType::TYPE_TILE_GRAPH},
            {    "SplitReshapeOpPVC2",     "SplitReshapeOpPVC2",   PassType::TYPE_TILE_GRAPH},
            {     "RemoveRedundentOp",      "RemoveRedundentOp",   PassType::TYPE_TILE_GRAPH},
            {        "GenerateMoveOp",         "GenerateMoveOp",   PassType::TYPE_TILE_GRAPH},
    });

    int h = 128, minusTwo = -2;
    Tensor output(DT_INT32, {h, h}, "output");
    Tensor idxs(DT_INT32, {h, h}, "idxs");
    Tensor keyStates(DT_INT32, {h, h}, "keyStates");

    std::string funcName = "ScatterUpdate";
    FUNCTION(funcName) {
        output = ScatterUpdate(output, idxs, keyStates, minusTwo);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
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
        DynParamInfo fakeParam = {2, 0, 0, DynParamInfoType::VALID_SHAPE, 0};
        subFunc.second->InsertDynParam("sym_32_dim_0", fakeParam);
        subFunc.second->InsertDynParam("sym_32_dim_1", fakeParam);
        subFunc.second->InsertDynParam("sym_38_dim_0", fakeParam);
        subFunc.second->InsertDynParam("sym_38_dim_1", fakeParam);
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
