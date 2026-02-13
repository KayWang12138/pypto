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
 * \file test_main_block_log.cpp
 * \brief Unit tests for main_block.cpp covering MACHINE_LOG calls at lines 61, 95
 */

#include <gtest/gtest.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk_log.h"

#define private public
#include "machine/host/main_block.h"
#undef private

using namespace npu::tile_fwk;

class TestMainBlockLog : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
    }
    void TearDown() override {}
};

// ===================================================================
// Tests for GetValidShapeFromCoa
// ===================================================================

// Covers Line 61: MACHINE_LOGW("argList is empty!")
// GetValidShapeFromCoa returns false when argList is empty
TEST_F(TestMainBlockLog, GetValidShapeFromCoa_EmptyArgList) {
    MainBlockCondBulider builder;
    std::vector<SymbolicScalar> emptyArgList;
    Shape shape;
    std::vector<SymbolicScalar> dynValidShape;

    bool result = builder.GetValidShapeFromCoa(emptyArgList, shape, dynValidShape);
    EXPECT_FALSE(result);
    EXPECT_TRUE(shape.empty());
    EXPECT_TRUE(dynValidShape.empty());
}

// ===================================================================
// Tests for CollectCallopMainBlockConds
// ===================================================================

// Covers Line 95: MACHINE_LOGW("get mainBlock flag false, op code %s, %s shape is %s, validShape is %s", ...)
// Operations have shapes but empty dynValidShape_ -> CheckShapeEquality returns false -> warning logged
TEST_F(TestMainBlockLog, CollectCallopMainBlockConds_ShapeMismatch) {
    config::SetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE, 1);

    int tileSize = 32;
    TileShape::Current().SetVecTile(tileSize, tileSize);
    int tensorDim = tileSize * 4;

    Tensor tensorA(DT_INT32, {tensorDim, tensorDim}, "A");
    Tensor tensorB(DT_INT32, {tensorDim, tensorDim}, "B");
    Tensor outputTensor(DT_INT32, {tensorDim, tensorDim}, "O");

    FUNCTION("test_mainblock_log", {tensorA, tensorB}, {outputTensor}) {
        LOOP("s0", FunctionType::DYNAMIC_LOOP, rowIdx, LoopRange(0x4)) {
            LOOP("s1", FunctionType::DYNAMIC_LOOP, colIdx, LoopRange(0x4)) {
                Tensor tileA = View(tensorA, {tileSize, tileSize}, {rowIdx * tileSize, colIdx * tileSize});
                Tensor tileB = View(tensorB, {tileSize, tileSize}, {rowIdx * tileSize, colIdx * tileSize});
                Tensor tileSum = Add(tileA, tileB);
                Assemble(tileSum, {rowIdx * tileSize, colIdx * tileSize}, outputTensor);
            }
        }
    }

    Function *func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);

    MainBlockCondBulider builder;
    // Operations have shapes like {32,32} but dynValidShape_ is empty by default
    // -> CheckShapeEquality returns false -> line 95 MACHINE_LOGW is hit
    builder.CollectCallopMainBlockConds(func);

    // Verify that a false condition was added (due to shape mismatch)
    EXPECT_FALSE(builder.GetCondGroup().empty());
}
