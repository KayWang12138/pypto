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
 * \file test_codegen_dyn_transpose_data_move.cpp
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
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenDynTransposeDataMove : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_TENSOR_GRAPH, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

void TestTransposeDataMoveBody(int dim = 3) {
    std::vector<int64_t> shape = {64, 64, 64};
    if (dim == SHAPE_DIM4) {
        shape = {64, 64, 64, 64};
    }
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

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
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(DataType::DT_FP32, shape, "TransposeDataMove", 123);
    std::vector<int64_t> offset = {0, 0, 0};
    if (dim == SHAPE_DIM4) {
        offset = {0, 0, 0, 0};
    }
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

    auto &op = function->AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {localTensor}, {ddrTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
    auto to_offset = OpImmediate::Specified({0, 0, 0});
    if (dim == SHAPE_DIM4) {
        to_offset = OpImmediate::Specified({0, 0, 0, 0});
    }
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, to_offset, shapeImme, shapeImme));
    op.SetOOpAttrOffset(0, 0);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    SymbolManager memAlloc;
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, memAlloc);
    CodeGenOpCloudNPU cop(memAlloc, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;

    cop.Init(op);
    cop.GenOpCode();
}

TEST_F(TestCodegenDynTransposeDataMove, TransposeDataMoveDim3) {
    TestTransposeDataMoveBody();
}

TEST_F(TestCodegenDynTransposeDataMove, TransposeDataMoveDim4) {
    TestTransposeDataMoveBody(SHAPE_DIM4);
}
} // namespace npu::tile_fwk
