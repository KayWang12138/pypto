/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify
 * it under the terms and conditions of CANN Open Software License
 * Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file
 * except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES
 * OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
 * TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
 * PURPOSE.
 * See LICENSE in the root of the software repository for the full
 * text of the License.
 */

/*!
 * \file test_block_ops.cpp
 * \brief Unit tests for block operator type deduction via OpRegistry
 */

#include "gtest/gtest.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class BlockOpsTest : public testing::Test {
protected:
    OpRegistry &registry_ = OpRegistry::GetInstance();
    Span span_ = Span::Unknown();

    VarPtr MakeTileVar(const std::string &name, std::vector<int64_t> dims, DataType dtype = DataType::FP32) {
        std::vector<ExprPtr> shape;
        for (auto d : dims) {
            shape.push_back(std::make_shared<ConstInt>(d, DataType::INT64, span_));
        }
        auto tt = std::make_shared<TileType>(shape, dtype);
        return std::make_shared<Var>(name, tt, span_);
    }

    VarPtr MakeTensorVar(const std::string &name, std::vector<int64_t> dims, DataType dtype = DataType::FP32) {
        std::vector<ExprPtr> shape;
        for (auto d : dims) {
            shape.push_back(std::make_shared<ConstInt>(d, DataType::INT64, span_));
        }
        auto tt = std::make_shared<TensorType>(shape, dtype);
        return std::make_shared<Var>(name, tt, span_);
    }

    VarPtr MakeScalarVar(const std::string &name, DataType dtype = DataType::FP32) {
        auto st = std::make_shared<ScalarType>(dtype);
        return std::make_shared<Var>(name, st, span_);
    }

    ExprPtr MakeConstInt(int64_t val, DataType dt = DataType::INT32) {
        return std::make_shared<ConstInt>(val, dt, span_);
    }

    ExprPtr MakeShapeTuple(std::vector<int64_t> dims, DataType dtype = DataType::INT64) {
        std::vector<ExprPtr> elems;
        for (auto d : dims) {
            elems.push_back(std::make_shared<ConstInt>(d, dtype, span_));
        }
        return std::make_shared<MakeTuple>(elems, span_);
    }
};

// ================================================================
// Block Elementwise Binary Ops
// ================================================================

TEST_F(BlockOpsTest, TestBlockAddCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeTileVar("b", {4, 8});
    auto call = registry_.Create("block.add", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockSubCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeTileVar("b", {4, 8});
    auto call = registry_.Create("block.sub", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockMulCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeTileVar("b", {4, 8});
    auto call = registry_.Create("block.mul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockDivCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeTileVar("b", {4, 8});
    auto call = registry_.Create("block.div", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockMaximumCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeTileVar("b", {4, 8});
    auto call = registry_.Create("block.maximum", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockMinimumCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeTileVar("b", {4, 8});
    auto call = registry_.Create("block.minimum", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

// ================================================================
// Block Scalar Binary Ops
// ================================================================

TEST_F(BlockOpsTest, TestBlockMulsCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeScalarVar("b", DataType::FP32);
    auto call = registry_.Create("block.muls", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsTest, TestBlockAddsCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeScalarVar("b", DataType::FP32);
    auto call = registry_.Create("block.adds", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockDivsCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeScalarVar("b", DataType::FP32);
    auto call = registry_.Create("block.divs", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockSubsCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeScalarVar("b", DataType::FP32);
    auto call = registry_.Create("block.subs", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

// ================================================================
// Block Comparison Ops
// ================================================================

TEST_F(BlockOpsTest, TestBlockCmpCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeTileVar("b", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cmp_type", std::any(0)}
    };
    auto call = registry_.Create("block.cmp", {a, b}, kwargs, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockCmpsCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeScalarVar("b", DataType::FP32);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cmp_type", std::any(0)}
    };
    auto call = registry_.Create("block.cmps", {a, b}, kwargs, span_);
    ASSERT_NE(call, nullptr);
}

// ================================================================
// Block Col Expand / Expands Ops
// ================================================================

TEST_F(BlockOpsTest, TestBlockColExpandCreate) {
    auto target = MakeTileVar("target", {4, 8});
    auto col = MakeTileVar("col", {1, 8});
    auto call = registry_.Create("block.col_expand", {target, col}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockColExpandMulCreate) {
    auto target = MakeTileVar("target", {4, 8});
    auto col = MakeTileVar("col", {1, 8});
    auto call = registry_.Create("block.col_expand_mul", {target, col}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockColExpandDivCreate) {
    auto target = MakeTileVar("target", {4, 8});
    auto col = MakeTileVar("col", {1, 8});
    auto call = registry_.Create("block.col_expand_div", {target, col}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockColExpandSubCreate) {
    auto target = MakeTileVar("target", {4, 8});
    auto col = MakeTileVar("col", {1, 8});
    auto call = registry_.Create("block.col_expand_sub", {target, col}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockExpandsCreate) {
    auto target = MakeTileVar("target", {4, 8});
    auto scalar = MakeScalarVar("s", DataType::FP32);
    auto call = registry_.Create("block.expands", {target, scalar}, span_);
    ASSERT_NE(call, nullptr);
}

// ================================================================
// Block Memory Ops
// ================================================================

TEST_F(BlockOpsTest, TestBlockGetBlockIdxCreate) {
    auto call = registry_.Create("block.get_block_idx", {}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsTest, TestBlockLoadCreate) {
    auto tensor = MakeTensorVar("t", {64, 128});
    auto rowOff = MakeConstInt(0);
    auto colOff = MakeConstInt(0);
    auto height = MakeConstInt(4);
    auto width = MakeConstInt(8);
    auto call = registry_.Create("block.load", {tensor, rowOff, colOff, height, width}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockStoreCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    auto rowOff = MakeConstInt(0);
    auto colOff = MakeConstInt(0);
    auto height = MakeConstInt(4);
    auto width = MakeConstInt(8);
    auto out = MakeTensorVar("out", {64, 128});
    auto call = registry_.Create("block.store", {tile, rowOff, colOff, height, width, out}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockL0cStoreCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    auto rowOff = MakeConstInt(0);
    auto colOff = MakeConstInt(0);
    auto height = MakeConstInt(4);
    auto width = MakeConstInt(8);
    auto out = MakeTensorVar("out", {64, 128});
    auto call = registry_.Create("block.l0c_store", {tile, rowOff, colOff, height, width, out}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockMoveCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    auto call = registry_.Create("block.move", {tile}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsTest, TestBlockMoveWithTranspose) {
    auto tile = MakeTileVar("tile", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"transpose", std::any(true)}
    };
    auto call = registry_.Create("block.move", {tile}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockAllocCreate) {
    auto memSpace = MakeConstInt(0);
    auto addr = MakeConstInt(0);
    auto size = MakeConstInt(1024);
    auto id = MakeConstInt(0);
    auto call = registry_.Create("block.alloc", {memSpace, addr, size, id}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockZerosCreate) {
    auto h = MakeConstInt(4);
    auto w = MakeConstInt(8);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", std::any(0x34)}
    };
    auto call = registry_.Create("block.zeros", {h, w}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

// ================================================================
// Block Reduction Ops
// ================================================================

TEST_F(BlockOpsTest, TestBlockSumCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {   "axis",    std::any(1)},
        {"keepdim", std::any(true)}
    };
    auto call = registry_.Create("block.sum", {tile}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsTest, TestBlockMaxCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {   "axis",     std::any(0)},
        {"keepdim", std::any(false)}
    };
    auto call = registry_.Create("block.max", {tile}, kwargs, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockMinCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {   "axis",     std::any(1)},
        {"keepdim", std::any(false)}
    };
    auto call = registry_.Create("block.min", {tile}, kwargs, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockSumKeepDim) {
    auto tile = MakeTileVar("tile", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {   "axis",    std::any(1)},
        {"keepdim", std::any(true)}
    };
    auto call = registry_.Create("block.sum", {tile}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockSumNoKeepDim) {
    auto tile = MakeTileVar("tile", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {   "axis",     std::any(1)},
        {"keepdim", std::any(false)}
    };
    auto call = registry_.Create("block.sum", {tile}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 1);
}

TEST_F(BlockOpsTest, TestBlockRowMaxCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    auto call = registry_.Create("block.row_max", {tile}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockRowSumCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    auto call = registry_.Create("block.row_sum", {tile}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockRowMinCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    auto call = registry_.Create("block.row_min", {tile}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockSumNegativeAxis) {
    auto tile = MakeTileVar("tile", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {   "axis",    std::any(-1)},
        {"keepdim", std::any(false)}
    };
    auto call = registry_.Create("block.sum", {tile}, kwargs, span_);
    ASSERT_NE(call, nullptr);
}

// ================================================================
// Block Batch MatMul Op
// ================================================================

TEST_F(BlockOpsTest, TestBlockBatchMatMul2D) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeTileVar("b", {8, 16});
    auto call = registry_.Create("block.batch_matmul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockBatchMatMulBatched) {
    auto a = MakeTileVar("a", {2, 4, 8});
    auto b = MakeTileVar("b", {2, 8, 16});
    auto call = registry_.Create("block.batch_matmul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 3);
}

// ================================================================
// IsRegistered Tests
// ================================================================

// ================================================================
// Block Unary Ops
// ================================================================

TEST_F(BlockOpsTest, TestBlockNegCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto call = registry_.Create("block.neg", {a}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
    ASSERT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsTest, TestBlockExpCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto call = registry_.Create("block.exp", {a}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsTest, TestBlockRecipCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto call = registry_.Create("block.recip", {a}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsTest, TestBlockSqrtCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto call = registry_.Create("block.sqrt", {a}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockRsqrtCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto call = registry_.Create("block.rsqrt", {a}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockLogCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto call = registry_.Create("block.log", {a}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockAbsCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto call = registry_.Create("block.abs", {a}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockReluCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto call = registry_.Create("block.relu", {a}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockCastCreate) {
    auto a = MakeTileVar("a", {4, 8}, DataType::FP32);
    int fp16Code = static_cast<int>(DataType::FP16.Code());
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"target_dtype", std::any(fp16Code)}
    };
    auto call = registry_.Create("block.cast", {a}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsTest, TestBlockUnaryPreservesShape) {
    auto a = MakeTileVar("a", {16, 32}, DataType::FP16);
    auto call = registry_.Create("block.neg", {a}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::FP16);
    ASSERT_EQ(rt->shape_.size(), 2);
}

// ================================================================
// Block MatMul Ops
// ================================================================

TEST_F(BlockOpsTest, TestBlockMatMulCreate) {
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeTileVar("b", {8, 16});
    auto call = registry_.Create("block.matmul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockMatMulAccCreate) {
    auto acc = MakeTileVar("acc", {4, 16});
    auto a = MakeTileVar("a", {4, 8});
    auto b = MakeTileVar("b", {8, 16});
    auto call = registry_.Create("block.matmul_acc", {acc, a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockMatMulDtypePromotion) {
    auto a = MakeTileVar("a", {4, 8}, DataType::FP16);
    auto b = MakeTileVar("b", {8, 16}, DataType::FP16);
    auto call = registry_.Create("block.matmul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

// ================================================================
// Block Row Expand (Broadcast) Ops
// ================================================================

TEST_F(BlockOpsTest, TestBlockRowExpandSubCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    auto rowVec = MakeTileVar("row_vec", {4, 1});
    auto call = registry_.Create("block.row_expand_sub", {tile, rowVec}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockRowExpandDivCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    auto rowVec = MakeTileVar("row_vec", {4, 1});
    auto call = registry_.Create("block.row_expand_div", {tile, rowVec}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockRowExpandMulCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    auto rowVec = MakeTileVar("row_vec", {4, 1});
    auto call = registry_.Create("block.row_expand_mul", {tile, rowVec}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockRowExpandAddCreate) {
    auto tile = MakeTileVar("tile", {4, 8});
    auto rowVec = MakeTileVar("row_vec", {4, 1});
    auto call = registry_.Create("block.row_expand_add", {tile, rowVec}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

// ================================================================
// Block Transform Ops
// ================================================================

TEST_F(BlockOpsTest, TestBlockViewCreate) {
    auto input = MakeTileVar("input", {16, 32});
    auto shape = MakeShapeTuple({4, 8});
    auto offset = MakeShapeTuple({0, 0});
    auto call = registry_.Create("block.view", {input, shape, offset}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockReshapeCreate) {
    auto input = MakeTileVar("input", {4, 8});
    auto shape = MakeShapeTuple({2, 16});
    auto call = registry_.Create("block.reshape", {input, shape}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockTransposeCreate) {
    auto input = MakeTileVar("input", {4, 8});
    auto axis1 = std::make_shared<ConstInt>(0, DataType::INT32, span_);
    auto axis2 = std::make_shared<ConstInt>(1, DataType::INT32, span_);
    auto call = registry_.Create("block.transpose", {input, axis1, axis2}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockTransposeNegativeAxis) {
    auto input = MakeTileVar("input", {4, 8});
    auto axis1 = std::make_shared<ConstInt>(0, DataType::INT32, span_);
    auto axis2 = std::make_shared<ConstInt>(-1, DataType::INT32, span_);
    auto call = registry_.Create("block.transpose", {input, axis1, axis2}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestAllBlockOpsRegistered) {
    ASSERT_TRUE(registry_.IsRegistered("block.add"));
    ASSERT_TRUE(registry_.IsRegistered("block.sub"));
    ASSERT_TRUE(registry_.IsRegistered("block.mul"));
    ASSERT_TRUE(registry_.IsRegistered("block.div"));
    ASSERT_TRUE(registry_.IsRegistered("block.maximum"));
    ASSERT_TRUE(registry_.IsRegistered("block.minimum"));
    ASSERT_TRUE(registry_.IsRegistered("block.muls"));
    ASSERT_TRUE(registry_.IsRegistered("block.adds"));
    ASSERT_TRUE(registry_.IsRegistered("block.divs"));
    ASSERT_TRUE(registry_.IsRegistered("block.subs"));
    ASSERT_TRUE(registry_.IsRegistered("block.cmp"));
    ASSERT_TRUE(registry_.IsRegistered("block.cmps"));
    ASSERT_TRUE(registry_.IsRegistered("block.col_expand"));
    ASSERT_TRUE(registry_.IsRegistered("block.expands"));
    ASSERT_TRUE(registry_.IsRegistered("block.get_block_idx"));
    ASSERT_TRUE(registry_.IsRegistered("block.load"));
    ASSERT_TRUE(registry_.IsRegistered("block.store"));
    ASSERT_TRUE(registry_.IsRegistered("block.move"));
    ASSERT_TRUE(registry_.IsRegistered("block.alloc"));
    ASSERT_TRUE(registry_.IsRegistered("block.zeros"));
    ASSERT_TRUE(registry_.IsRegistered("block.sum"));
    ASSERT_TRUE(registry_.IsRegistered("block.max"));
    ASSERT_TRUE(registry_.IsRegistered("block.min"));
    ASSERT_TRUE(registry_.IsRegistered("block.row_max"));
    ASSERT_TRUE(registry_.IsRegistered("block.row_sum"));
    ASSERT_TRUE(registry_.IsRegistered("block.row_min"));
    ASSERT_TRUE(registry_.IsRegistered("block.batch_matmul"));
    // Unary ops
    ASSERT_TRUE(registry_.IsRegistered("block.neg"));
    ASSERT_TRUE(registry_.IsRegistered("block.exp"));
    ASSERT_TRUE(registry_.IsRegistered("block.recip"));
    ASSERT_TRUE(registry_.IsRegistered("block.sqrt"));
    ASSERT_TRUE(registry_.IsRegistered("block.rsqrt"));
    ASSERT_TRUE(registry_.IsRegistered("block.cast"));
    ASSERT_TRUE(registry_.IsRegistered("block.log"));
    ASSERT_TRUE(registry_.IsRegistered("block.abs"));
    ASSERT_TRUE(registry_.IsRegistered("block.relu"));
    // MatMul ops
    ASSERT_TRUE(registry_.IsRegistered("block.matmul"));
    ASSERT_TRUE(registry_.IsRegistered("block.matmul_acc"));
    // Broadcast ops
    ASSERT_TRUE(registry_.IsRegistered("block.row_expand_sub"));
    ASSERT_TRUE(registry_.IsRegistered("block.row_expand_div"));
    ASSERT_TRUE(registry_.IsRegistered("block.row_expand_mul"));
    ASSERT_TRUE(registry_.IsRegistered("block.row_expand_add"));
    // Transform ops
    ASSERT_TRUE(registry_.IsRegistered("block.view"));
    ASSERT_TRUE(registry_.IsRegistered("block.reshape"));
    ASSERT_TRUE(registry_.IsRegistered("block.transpose"));
    // Col expand ops
    ASSERT_TRUE(registry_.IsRegistered("block.col_expand_mul"));
    ASSERT_TRUE(registry_.IsRegistered("block.col_expand_div"));
    ASSERT_TRUE(registry_.IsRegistered("block.col_expand_sub"));
}

// ================================================================
// Block Transform Ops - Runtime Tuple and Edge Case Tests
// ================================================================

TEST_F(BlockOpsTest, TestBlockViewRuntimeShape) {
    auto input = MakeTileVar("input", {16, 32});
    // Runtime shape tuple (Var with TupleType instead of MakeTuple)
    auto tupleType = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64), std::make_shared<ScalarType>(DataType::INT64)});
    auto shapeVar = std::make_shared<Var>("shape", tupleType, span_);
    auto offset = MakeShapeTuple({0, 0});
    auto call = registry_.Create("block.view", {input, shapeVar, offset}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockReshapeRuntimeShape) {
    auto input = MakeTileVar("input", {4, 8});
    // Runtime shape tuple (Var with TupleType instead of MakeTuple)
    auto tupleType = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64), std::make_shared<ScalarType>(DataType::INT64)});
    auto shapeVar = std::make_shared<Var>("shape", tupleType, span_);
    auto call = registry_.Create("block.reshape", {input, shapeVar}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(BlockOpsTest, TestBlockReshapeStaticProductCheck) {
    // Reshape to a shape with same total number of elements
    auto input = MakeTileVar("input", {4, 8});
    std::vector<ExprPtr> shapeElems = {std::make_shared<ConstInt>(32, DataType::INT64, span_)};
    auto shape = std::make_shared<MakeTuple>(shapeElems, span_);
    auto call = registry_.Create("block.reshape", {input, shape}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 1);
}

TEST_F(BlockOpsTest, TestBlockReshape3D) {
    auto input = MakeTileVar("input", {4, 8});
    auto shape = MakeShapeTuple({2, 2, 8});
    auto call = registry_.Create("block.reshape", {input, shape}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 3);
}

TEST_F(BlockOpsTest, TestBlockTranspose3D) {
    auto input = MakeTileVar("input", {2, 4, 8});
    auto axis1 = std::make_shared<ConstInt>(0, DataType::INT32, span_);
    auto axis2 = std::make_shared<ConstInt>(2, DataType::INT32, span_);
    auto call = registry_.Create("block.transpose", {input, axis1, axis2}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 3);
}

TEST_F(BlockOpsTest, TestBlockViewUINT64Shape) {
    auto input = MakeTileVar("input", {16, 32});
    auto shape = MakeShapeTuple({4, 8}, DataType::UINT64);
    auto offset = MakeShapeTuple({0, 0}, DataType::UINT64);
    auto call = registry_.Create("block.view", {input, shape, offset}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockReshapeUINT64Shape) {
    auto input = MakeTileVar("input", {4, 8});
    auto shape = MakeShapeTuple({2, 16}, DataType::UINT64);
    auto call = registry_.Create("block.reshape", {input, shape}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(BlockOpsTest, TestBlockViewPreservesDtype) {
    auto input = MakeTileVar("input", {16, 32}, DataType::FP16);
    auto shape = MakeShapeTuple({4, 8});
    auto offset = MakeShapeTuple({0, 0});
    auto call = registry_.Create("block.view", {input, shape, offset}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsTest, TestBlockReshapePreservesDtype) {
    auto input = MakeTileVar("input", {4, 8}, DataType::FP16);
    auto shape = MakeShapeTuple({2, 16});
    auto call = registry_.Create("block.reshape", {input, shape}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsTest, TestBlockTransposePreservesDtype) {
    auto input = MakeTileVar("input", {4, 8}, DataType::FP16);
    auto axis1 = std::make_shared<ConstInt>(0, DataType::INT32, span_);
    auto axis2 = std::make_shared<ConstInt>(1, DataType::INT32, span_);
    auto call = registry_.Create("block.transpose", {input, axis1, axis2}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::FP16);
}

} // namespace ir
} // namespace pypto
