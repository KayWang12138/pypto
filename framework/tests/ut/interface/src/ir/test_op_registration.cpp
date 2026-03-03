/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_op_registration.cpp
 * \brief Unit tests for operator registration (block_ops, tensor_ops, sync_ops,
 *        op_registry, type_inference)
 */

#include "gtest/gtest.h"

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

static Span TestSpan() {
    return Span("test.py", 1, 0);
}

// ============================================================================
// OpRegistry Tests (op_registry.cpp)
// ============================================================================

TEST(OpRegistryExtraTest, GetInstance) {
    auto &registry = OpRegistry::GetInstance();
    (void)registry;
    ASSERT_TRUE(true); // Singleton should exist
}

TEST(OpRegistryExtraTest, BlockAddRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("block.add"));
}

TEST(OpRegistryExtraTest, NonExistentOp) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_FALSE(registry.IsRegistered("nonexistent.op"));
}

// ============================================================================
// Block Ops Elementwise Registration Tests (block_ops/elementwise.cpp)
// ============================================================================

class BlockOpsElementwiseTest : public ::testing::TestWithParam<std::string> {};

TEST_P(BlockOpsElementwiseTest, IsRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered(GetParam()));
    const auto &entry = registry.GetEntry(GetParam());
    ASSERT_FALSE(entry.GetOpCategory().empty());
}

INSTANTIATE_TEST_SUITE_P(BlockElementwise, BlockOpsElementwiseTest,
    ::testing::Values("block.mul", "block.add", "block.div", "block.sub", "block.maximum", "block.minimum",
        "block.muls", "block.adds", "block.divs", "block.subs", "block.cmp", "block.cmps", "block.col_expand",
        "block.col_expand_mul", "block.col_expand_div", "block.col_expand_sub", "block.expands"));

// ============================================================================
// Block Ops Unary Registration Tests (block_ops/unary.cpp)
// ============================================================================

class BlockOpsUnaryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(BlockOpsUnaryTest, IsRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(BlockUnary, BlockOpsUnaryTest,
    ::testing::Values("block.neg", "block.exp", "block.recip", "block.sqrt", "block.rsqrt", "block.cast", "block.log",
        "block.abs", "block.relu"));

// ============================================================================
// Block Ops Matmul Registration Tests (block_ops/matmul.cpp)
// ============================================================================

TEST(BlockOpsMatmulTest, MatmulRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("block.matmul"));
    ASSERT_TRUE(registry.IsRegistered("block.matmul_acc"));
}

// ============================================================================
// Block Ops Batch Matmul Registration Tests (block_ops/batch_matmul.cpp)
// ============================================================================

TEST(BlockOpsBatchMatmulTest, BatchMatmulRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("block.batch_matmul"));
}

// ============================================================================
// Block Ops Reduction Registration Tests (block_ops/reduction.cpp)
// ============================================================================

class BlockOpsReductionTest : public ::testing::TestWithParam<std::string> {};

TEST_P(BlockOpsReductionTest, IsRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(BlockReduction, BlockOpsReductionTest,
    ::testing::Values("block.sum", "block.max", "block.min", "block.row_max", "block.row_sum", "block.row_min"));

// ============================================================================
// Block Ops Broadcast Registration Tests (block_ops/broadcast.cpp)
// ============================================================================

class BlockOpsBroadcastTest : public ::testing::TestWithParam<std::string> {};

TEST_P(BlockOpsBroadcastTest, IsRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(BlockBroadcast, BlockOpsBroadcastTest,
    ::testing::Values("block.row_expand_sub", "block.row_expand_div", "block.row_expand_mul", "block.row_expand_add"));

// ============================================================================
// Block Ops Memory Registration Tests (block_ops/memory.cpp)
// ============================================================================

class BlockOpsMemoryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(BlockOpsMemoryTest, IsRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(BlockMemory, BlockOpsMemoryTest,
    ::testing::Values("block.get_block_idx", "block.load", "block.store", "block.l0c_store", "block.move",
        "block.alloc", "block.zeros"));

// ============================================================================
// Block Ops Transform Registration Tests (block_ops/transform.cpp)
// ============================================================================

class BlockOpsTransformTest : public ::testing::TestWithParam<std::string> {};

TEST_P(BlockOpsTransformTest, IsRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    BlockTransform, BlockOpsTransformTest, ::testing::Values("block.view", "block.reshape", "block.transpose"));

// ============================================================================
// Tensor Ops Elementwise Registration Tests (tensor_ops/elementwise.cpp)
// ============================================================================

class TensorOpsElementwiseTest : public ::testing::TestWithParam<std::string> {};

TEST_P(TensorOpsElementwiseTest, IsRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(TensorElementwise, TensorOpsElementwiseTest,
    ::testing::Values("tensor.add", "tensor.add_scalar", "tensor.sub", "tensor.sub_scalar", "tensor.mul",
        "tensor.mul_scalar", "tensor.div", "tensor.div_scalar", "tensor.maximum"));

// ============================================================================
// Tensor Ops Unary Registration Tests (tensor_ops/unary.cpp)
// ============================================================================

TEST(TensorOpsUnaryTest, ExpRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("tensor.exp"));
}

TEST(TensorOpsUnaryTest, CastRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("tensor.cast"));
}

// ============================================================================
// Tensor Ops Matmul Registration Tests (tensor_ops/matmul.cpp)
// ============================================================================

TEST(TensorOpsMatmulTest, MatmulRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("tensor.matmul"));
}

// ============================================================================
// Tensor Ops Memory Registration Tests (tensor_ops/memory.cpp)
// ============================================================================

class TensorOpsMemoryTest : public ::testing::TestWithParam<std::string> {};

TEST_P(TensorOpsMemoryTest, IsRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    TensorMemory, TensorOpsMemoryTest, ::testing::Values("tensor.create", "tensor.view", "tensor.assemble"));

// ============================================================================
// Tensor Ops Reduction Registration Tests (tensor_ops/reduction.cpp)
// ============================================================================

TEST(TensorOpsReductionTest, RowMaxRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("tensor.row_max"));
}

TEST(TensorOpsReductionTest, RowSumRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("tensor.row_sum"));
}

// ============================================================================
// Tensor Ops Transform Registration Tests (tensor_ops/transform.cpp)
// ============================================================================

TEST(TensorOpsTransformTest, ReshapeRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("tensor.reshape"));
}

TEST(TensorOpsTransformTest, TransposeRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("tensor.transpose"));
}

// ============================================================================
// Op Entry Detail Tests
// ============================================================================

TEST(OpEntryTest, BlockAddHasDescription) {
    auto &registry = OpRegistry::GetInstance();
    const auto &entry = registry.GetEntry("block.add");
    ASSERT_FALSE(entry.GetDescription().empty());
}

TEST(OpEntryTest, BlockAddHasCategory) {
    auto &registry = OpRegistry::GetInstance();
    const auto &entry = registry.GetEntry("block.add");
    ASSERT_FALSE(entry.GetOpCategory().empty());
}

TEST(OpEntryTest, GetOpReturnsOp) {
    auto &registry = OpRegistry::GetInstance();
    auto op = registry.GetOp("block.add");
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->name_, "block.add");
}

TEST(OpEntryTest, SyncSrcHasAttributes) {
    auto &registry = OpRegistry::GetInstance();
    auto op = registry.GetOp("system.sync_src");
    ASSERT_NE(op, nullptr);
    ASSERT_TRUE(op->HasAttr("set_pipe"));
    ASSERT_TRUE(op->HasAttr("wait_pipe"));
    ASSERT_TRUE(op->HasAttr("event_id"));
}

TEST(OpEntryTest, BlockCastHasAttributes) {
    auto &registry = OpRegistry::GetInstance();
    auto op = registry.GetOp("block.cast");
    ASSERT_NE(op, nullptr);
    ASSERT_TRUE(op->HasAttr("target_dtype"));
}

// ============================================================================
// Type Inference Tests (type_inference.cpp)
// ============================================================================

TEST(TypeInferenceUtilsTest, PromoteDataTypesSameType) {
    auto result = PromoteDataTypes(DataType::INT32, DataType::INT32);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value(), DataType::INT32);
}

TEST(TypeInferenceUtilsTest, PromoteDataTypesIntToFloat) {
    auto result = PromoteDataTypes(DataType::INT32, DataType::FP32);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value(), DataType::FP32);
}

TEST(TypeInferenceUtilsTest, BroadcastShapesSameShape) {
    std::vector<ExprPtr> shape1 = {std::make_shared<ConstInt>(4, DataType::INT64, TestSpan()),
        std::make_shared<ConstInt>(8, DataType::INT64, TestSpan())};
    std::vector<ExprPtr> shape2 = {std::make_shared<ConstInt>(4, DataType::INT64, TestSpan()),
        std::make_shared<ConstInt>(8, DataType::INT64, TestSpan())};
    auto result = BroadcastShapes(shape1, shape2);
    ASSERT_TRUE(result.success);
}

TEST(TypeInferenceUtilsTest, BroadcastShapesWithOne) {
    std::vector<ExprPtr> shape1 = {std::make_shared<ConstInt>(4, DataType::INT64, TestSpan()),
        std::make_shared<ConstInt>(1, DataType::INT64, TestSpan())};
    std::vector<ExprPtr> shape2 = {std::make_shared<ConstInt>(4, DataType::INT64, TestSpan()),
        std::make_shared<ConstInt>(8, DataType::INT64, TestSpan())};
    auto result = BroadcastShapes(shape1, shape2);
    ASSERT_TRUE(result.success);
}

TEST(TypeInferenceUtilsTest, FormatShapeEmpty) {
    std::vector<ExprPtr> shape;
    auto str = FormatShape(shape);
    ASSERT_FALSE(str.empty());
}

TEST(TypeInferenceUtilsTest, FormatShapeWithDims) {
    std::vector<ExprPtr> shape = {std::make_shared<ConstInt>(4, DataType::INT64, TestSpan()),
        std::make_shared<ConstInt>(8, DataType::INT64, TestSpan())};
    auto str = FormatShape(shape);
    ASSERT_FALSE(str.empty());
}

// ============================================================================
// Op Create and Type Deduction Tests
// ============================================================================

// Helper to create a TensorType Var for tensor op tests
static ExprPtr MakeTensorVar(const std::string &name, const std::vector<int64_t> &shape, DataType dtype) {
    auto span = TestSpan();
    std::vector<ExprPtr> shapeExprs;
    for (auto dim : shape) {
        shapeExprs.push_back(std::make_shared<ConstInt>(dim, DataType::INT64, span));
    }
    auto type = std::make_shared<TensorType>(shapeExprs, dtype);
    return std::make_shared<Var>(name, type, span);
}

// Helper to create a TileType Var for block op tests
static ExprPtr MakeTileVar(const std::string &name, const std::vector<int64_t> &shape, DataType dtype) {
    auto span = TestSpan();
    std::vector<ExprPtr> shapeExprs;
    for (auto dim : shape) {
        shapeExprs.push_back(std::make_shared<ConstInt>(dim, DataType::INT64, span));
    }
    auto type = std::make_shared<TileType>(shapeExprs, dtype);
    return std::make_shared<Var>(name, type, span);
}

// ---- Tensor Op Create Tests (exercises type deduction in tensor_ops/*.cpp) ----

TEST(OpCreateTest, CreateTensorAdd) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    auto b = MakeTensorVar("b", {4, 8}, DataType::FP32);
    std::vector<ExprPtr> args = {a, b};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.add", args, kwargs, span);
    ASSERT_NE(call, nullptr);
    ASSERT_NE(call->GetType(), nullptr);
}

TEST(OpCreateTest, CreateTensorSub) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    auto b = MakeTensorVar("b", {4, 8}, DataType::FP32);
    std::vector<ExprPtr> args = {a, b};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.sub", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateTensorMul) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    auto b = MakeTensorVar("b", {4, 8}, DataType::FP32);
    std::vector<ExprPtr> args = {a, b};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.mul", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateTensorDiv) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    auto b = MakeTensorVar("b", {4, 8}, DataType::FP32);
    std::vector<ExprPtr> args = {a, b};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.div", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateTensorMaximum) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    auto b = MakeTensorVar("b", {4, 8}, DataType::FP32);
    std::vector<ExprPtr> args = {a, b};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.maximum", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateTensorExp) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    std::vector<ExprPtr> args = {a};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.exp", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateTensorCast) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    std::vector<ExprPtr> args = {a};
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"target_type", std::any(DataType::FP16)}
    };
    auto call = registry.Create("tensor.cast", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateTensorRowMax) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    std::vector<ExprPtr> args = {a};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.row_max", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateTensorRowSum) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    std::vector<ExprPtr> args = {a};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.row_sum", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateTensorMatmul) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP16);
    auto b = MakeTensorVar("b", {8, 16}, DataType::FP16);
    std::vector<ExprPtr> args = {a, b};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.matmul", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateTensorReshape) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    // tensor.reshape takes (input, shape_tuple) as positional args
    auto dim = std::make_shared<ConstInt>(32, DataType::INT64, span);
    auto shapeTuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim}, span);
    std::vector<ExprPtr> args = {a, shapeTuple};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.reshape", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateTensorTranspose) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    // tensor.transpose takes (input, axis1, axis2) as positional args
    auto axis1 = std::make_shared<ConstInt>(0, DataType::INT64, span);
    auto axis2 = std::make_shared<ConstInt>(1, DataType::INT64, span);
    std::vector<ExprPtr> args = {a, axis1, axis2};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("tensor.transpose", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

// ---- Block Op Create Tests (exercises type deduction in block_ops/*.cpp) ----

TEST(OpCreateTest, CreateBlockAdd) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTileVar("a", {16, 16}, DataType::FP16);
    auto b = MakeTileVar("b", {16, 16}, DataType::FP16);
    std::vector<ExprPtr> args = {a, b};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("block.add", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateBlockSub) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTileVar("a", {16, 16}, DataType::FP16);
    auto b = MakeTileVar("b", {16, 16}, DataType::FP16);
    std::vector<ExprPtr> args = {a, b};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("block.sub", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateBlockMul) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTileVar("a", {16, 16}, DataType::FP16);
    auto b = MakeTileVar("b", {16, 16}, DataType::FP16);
    std::vector<ExprPtr> args = {a, b};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("block.mul", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateBlockNeg) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTileVar("a", {16, 16}, DataType::FP16);
    std::vector<ExprPtr> args = {a};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("block.neg", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateBlockExp) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTileVar("a", {16, 16}, DataType::FP16);
    std::vector<ExprPtr> args = {a};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("block.exp", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateBlockSum) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTileVar("a", {16, 16}, DataType::FP16);
    std::vector<ExprPtr> args = {a};
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {   "axis",     std::any(0)},
        {"keepdim", std::any(false)}
    };
    auto call = registry.Create("block.sum", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateBlockMax) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTileVar("a", {16, 16}, DataType::FP16);
    std::vector<ExprPtr> args = {a};
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {   "axis",     std::any(0)},
        {"keepdim", std::any(false)}
    };
    auto call = registry.Create("block.max", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateBlockCast) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTileVar("a", {16, 16}, DataType::FP16);
    std::vector<ExprPtr> args = {a};
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"target_dtype", std::any(0)}
    };
    auto call = registry.Create("block.cast", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateBlockMuls) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    auto a = MakeTileVar("a", {16, 16}, DataType::FP16);
    auto scalar = std::make_shared<ConstFloat>(2.0, DataType::FP16, span);
    std::vector<ExprPtr> args = {a, scalar};
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto call = registry.Create("block.muls", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

// ---- Sync Op Create Tests (exercises type deduction in sync_ops/sync.cpp) ----

TEST(OpCreateTest, CreateSyncSrc) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    std::vector<ExprPtr> args;
    std::vector<std::pair<std::string, std::any>> kwargs = {
        { "set_pipe", std::any(1)},
        {"wait_pipe", std::any(2)},
        { "event_id", std::any(0)}
    };
    auto call = registry.Create("system.sync_src", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

TEST(OpCreateTest, CreateSyncDst) {
    auto &registry = OpRegistry::GetInstance();
    auto span = TestSpan();
    std::vector<ExprPtr> args;
    std::vector<std::pair<std::string, std::any>> kwargs = {
        { "set_pipe", std::any(1)},
        {"wait_pipe", std::any(2)},
        { "event_id", std::any(0)}
    };
    auto call = registry.Create("system.sync_dst", args, kwargs, span);
    ASSERT_NE(call, nullptr);
}

} // namespace ir
} // namespace pypto
