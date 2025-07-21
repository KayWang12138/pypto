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
 * \file test_codegen_spillout.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen_op.h"
#include "codegen/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"

using namespace npu::tile_fwk;

class TestCodegenSpillOut : public ::testing::Test {
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

TEST_F(TestCodegenSpillOut, UBSpillOut) {
    const std::vector<int> shape = {64, 64};
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
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(DataType::DT_FP32, shape, "UBSpillOut", SYMBOL_STACK_BASE);
    const std::vector<int> offset = {0, 0};

    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    auto ubTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    ubTensor->UpdateSubgraphID(0);
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    ubTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    ubTensor->SetMagic(3);
    ubTensor->SetAttr(OpAttributeKey::needAlloc, true);
    ubTensor->memorymap[0].memId = 0;
    ubTensor->memorymap[0].start = 0;
    ubTensor->memorymap[0].end = 0;

    auto &op = function->AddOperation(Opcode::OP_COPY_OUT, {ubTensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenExtraAlloc(memAlloc, ubTensor, op);
    CodeGenOpCloudNPU cop(memAlloc, function->GetTensorMap(), function->GetFunctionType());
    function->GetTensorMap().inverseMap_[ubTensor->GetMagic()] = ubTensor;

    cop.Init(op);
    cop.originShape[0] = shape;
    cop.originShape[1] = shape;

    cop.GenOpCode();
}

TEST_F(TestCodegenSpillOut, L1SpillOut) {
    const std::vector<int> shape = {64, 64};
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
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(DataType::DT_FP32, shape, "L1SpillOut", SYMBOL_STACK_BASE);
    const std::vector<int> offset = {0, 0};

    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    auto l1Tensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    l1Tensor->UpdateSubgraphID(0);
    l1Tensor->SetMemoryTypeOriginal(MemoryType::MEM_L1);
    l1Tensor->SetMemoryTypeToBe(MemoryType::MEM_L1);
    l1Tensor->SetMagic(3);
    l1Tensor->SetAttr(OpAttributeKey::needAlloc, true);
    l1Tensor->memorymap[0].memId = 0;
    l1Tensor->memorymap[0].start = 0;
    l1Tensor->memorymap[0].end = 0;

    auto &op = function->AddOperation(Opcode::OP_L1_COPY_OUT, {l1Tensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenExtraAlloc(memAlloc, l1Tensor, op);
    CodeGenOpCloudNPU cop(memAlloc, function->GetTensorMap(), function->GetFunctionType());
    function->GetTensorMap().inverseMap_[l1Tensor->GetMagic()] = l1Tensor;

    cop.Init(op);
    cop.originShape[0] = shape;
    cop.originShape[1] = shape;

    cop.GenOpCode();
}
