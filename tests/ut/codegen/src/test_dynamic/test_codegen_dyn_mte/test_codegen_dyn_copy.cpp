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
 * \file test_codegen_dyn_copy.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

constexpr const int dummyRawMagic = 123;

class TestCodegenDynCopy : public ::testing::Test {
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

std::string TestL0COutBody(bool isDynamicUnalign) {
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    Program::GetInstance().GetTileShape().SetVecTileShapes(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, FunctionType::STATIC, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    if (isDynamicUnalign) {
        ConfigManager::Instance().SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    }
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(DataType::DT_FP32, shape, "L0CToOut", dummyRawMagic);
    const std::vector<int64_t> offset = {0, 0};

    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    auto localTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    localTensor->UpdateSubgraphID(0);
    localTensor->SetMemoryTypeOriginal(MemoryType::MEM_L0C);
    localTensor->SetMemoryTypeToBe(MemoryType::MEM_L0C);
    localTensor->SetMagic(3);
    localTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localTensor->memorymap[0].memId = 0;
    localTensor->memorymap[0].start = 0;
    localTensor->memorymap[0].end = 0;
    if(isDynamicUnalign){
        std::vector<SymbolicScalar> dynValidShape = {64, 64};
        localTensor->UpdateDynValidShape(dynValidShape);
        ddrTensor->UpdateDynValidShape(dynValidShape);
    }
    auto &op = function->AddOperation(Opcode::OP_COPY_OUT, {localTensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L0C, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    op.SetOOpAttrOffset(0, 0);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    if(isDynamicUnalign){
        op.SetAttribute("op_attr_is_nz", 1);
    }

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, memAlloc);
    CodeGenOpCloudNPU cop(memAlloc, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;

    cop.Init(op);
    cop.originShape[0] = ToVecInt(shape);
    cop.originShape[1] = ToVecInt(shape);
    return cop.GenOpCode();
}

TEST_F(TestCodegenDynCopy, L0CToOut) {
    std::string res = TestL0COutBody(false);
    std::string expect =
        R"!!!(TileOp::DynL0CCopyOut<float, float, 64, 64, 64, 64>((__gm__ float*)GET_PARAM_ADDR(param, 0, 0), (__cc__ float*)L0C_S0_E0, GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L0CToOutUnalign) {
    std::string res = TestL0COutBody(true);
    std::string expect = R"!!!(TileOp::DynL0CCopyOut<float, float, false>((__gm__ float*)GET_PARAM_ADDR(param, 0, 0), (__cc__ float*)L0C_S0_E0, 64, 64, GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 1), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

std::string TestL1CopyInBody(bool isNz = false, int outerValueForNz = 0, int innerValueForNz = 0) {
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    Program::GetInstance().GetTileShape().SetVecTileShapes(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, FunctionType::STATIC, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(DataType::DT_FP32, shape, "L1CopyIn", dummyRawMagic);
    const std::vector<int64_t> offset = {0, 0};

    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    auto localTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    localTensor->UpdateSubgraphID(0);
    localTensor->SetMemoryTypeOriginal(MemoryType::MEM_L1);
    localTensor->SetMemoryTypeToBe(MemoryType::MEM_L1);
    localTensor->SetMagic(3);
    localTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localTensor->memorymap[0].memId = 0;
    localTensor->memorymap[0].start = 0;
    localTensor->memorymap[0].end = 0;

    auto &op = function->AddOperation(Opcode::OP_COPY_IN, {ddrTensor}, {localTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    op.SetIOpAttrOffset(0, 0);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    if (isNz) {
        op.SetAttribute(OP_ATTR_PREFIX + "is_nz", 1);
        op.SetAttribute(OP_ATTR_PREFIX + "outer_value", outerValueForNz);
        op.SetAttribute(OP_ATTR_PREFIX + "inner_value", innerValueForNz);
    }

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, memAlloc);
    CodeGenOpCloudNPU cop(memAlloc, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;

    cop.Init(op);
    cop.originShape[0] = ToVecInt(shape);
    cop.originShape[1] = ToVecInt(shape);

    return cop.GenOpCode();
}

TEST_F(TestCodegenDynCopy, L1CopyIn) {
    std::string res = TestL1CopyInBody();
    std::string expect =
        R"!!!(TileOp::DynL1CopyIn<float, float, 64, 64>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyInNZ) {
    std::string res = TestL1CopyInBody(true);
    std::string expect =
        R"!!!(TileOp::DynL1CopyInNZ2NZ<float, float, 64, 64>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 1), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyInNZWithValue) {
    std::string res = TestL1CopyInBody(true, 1, 1);
    std::string expect =
        R"!!!(TileOp::DynL1CopyInNZ2NZ<float, float, 64, 64>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 1, 1, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, UBCopyIn) {
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    Program::GetInstance().GetTileShape().SetVecTileShapes(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, FunctionType::STATIC, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(DataType::DT_FP32, shape, "L1CopyIn", dummyRawMagic);
    const std::vector<int64_t> offset = {0, 0};

    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    auto localTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    localTensor->UpdateSubgraphID(0);
    localTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    localTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    localTensor->SetMagic(3);
    localTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localTensor->memorymap[0].memId = 0;
    localTensor->memorymap[0].start = 0;
    localTensor->memorymap[0].end = 0;

    auto &op = function->AddOperation(Opcode::OP_COPY_IN, {ddrTensor}, {localTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    op.SetIOpAttrOffset(0, 0);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, memAlloc);
    CodeGenOpCloudNPU cop(memAlloc, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;

    cop.Init(op);
    cop.originShape[0] = ToVecInt(shape);
    cop.originShape[1] = ToVecInt(shape);

    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynUBCopyIn<float, 1, 1, 1, 64, 64, 1, 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), 1, 1, 1, GET_PARAM_RAWSHAPE_2(param, 0, 0), 0, 0, 0, GET_PARAM_OFFSET_2(param, 0, 0));
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
