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
 * \file test_tensor_ops.cpp
 * \brief Unit tests for tensor operator type deduction via OpRegistry
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

class TensorOpsTest : public testing::Test {
protected:
    OpRegistry &registry_ = OpRegistry::GetInstance();
    Span span_ = Span::Unknown();

    // Helper: create a Var with TensorType [M, N]
    VarPtr MakeTensorVar(const std::string &name, std::vector<int64_t> dims, DataType dtype = DataType::FP32) {
        std::vector<ExprPtr> shape;
        for (auto d : dims) {
            shape.push_back(std::make_shared<ConstInt>(d, DataType::INT64, span_));
        }
        auto tt = std::make_shared<TensorType>(shape, dtype);
        return std::make_shared<Var>(name, tt, span_);
    }

    // Helper: create a Var with ScalarType
    VarPtr MakeScalarVar(const std::string &name, DataType dtype = DataType::FP32) {
        auto st = std::make_shared<ScalarType>(dtype);
        return std::make_shared<Var>(name, st, span_);
    }

    // Helper: create a MakeTuple of ConstInt(INT64)
    ExprPtr MakeShapeTuple(std::vector<int64_t> dims) {
        std::vector<ExprPtr> elems;
        for (auto d : dims) {
            elems.push_back(std::make_shared<ConstInt>(d, DataType::INT64, span_));
        }
        return std::make_shared<MakeTuple>(elems, span_);
    }
};

// ================================================================
// Elementwise Binary Ops (tensor.add/sub/mul/div/maximum)
// ================================================================

TEST_F(TensorOpsTest, TestTensorAddCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeTensorVar("b", {4, 8});
    auto call = registry_.Create("tensor.add", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorSubCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeTensorVar("b", {4, 8});
    auto call = registry_.Create("tensor.sub", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(TensorOpsTest, TestTensorMulCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeTensorVar("b", {4, 8});
    auto call = registry_.Create("tensor.mul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(TensorOpsTest, TestTensorDivCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeTensorVar("b", {4, 8});
    auto call = registry_.Create("tensor.div", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(TensorOpsTest, TestTensorMaximumCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeTensorVar("b", {4, 8});
    auto call = registry_.Create("tensor.maximum", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(TensorOpsTest, TestTensorAddBroadcast) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeTensorVar("b", {8});
    auto call = registry_.Create("tensor.add", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

// ================================================================
// Elementwise Scalar Ops (tensor.add_scalar/sub_scalar/etc.)
// ================================================================

TEST_F(TensorOpsTest, TestTensorAddScalarCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeScalarVar("b", DataType::FP32);
    auto call = registry_.Create("tensor.add_scalar", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorSubScalarCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeScalarVar("b", DataType::FP32);
    auto call = registry_.Create("tensor.sub_scalar", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(TensorOpsTest, TestTensorMulScalarCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeScalarVar("b", DataType::FP32);
    auto call = registry_.Create("tensor.mul_scalar", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(TensorOpsTest, TestTensorDivScalarCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeScalarVar("b", DataType::FP32);
    auto call = registry_.Create("tensor.div_scalar", {a, b}, span_);
    ASSERT_NE(call, nullptr);
}

// ================================================================
// Unary Ops (tensor.exp, tensor.cast)
// ================================================================

TEST_F(TensorOpsTest, TestTensorExpCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    auto call = registry_.Create("tensor.exp", {a}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(TensorOpsTest, TestTensorExpIntPromotesToFloat) {
    auto a = MakeTensorVar("a", {4, 8}, DataType::INT32);
    auto call = registry_.Create("tensor.exp", {a}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(TensorOpsTest, TestTensorCastCreate) {
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"target_type", std::any(DataType::FP16)}
    };
    auto call = registry_.Create("tensor.cast", {a}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(TensorOpsTest, TestTensorCastWithIntKwarg) {
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP32);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"target_type", std::any(static_cast<int>(1))}
    };
    auto call = registry_.Create("tensor.cast", {a}, kwargs, span_);
    ASSERT_NE(call, nullptr);
}

// ================================================================
// MatMul Op (tensor.matmul)
// ================================================================

TEST_F(TensorOpsTest, TestTensorMatMul2D) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeTensorVar("b", {8, 16});
    auto call = registry_.Create("tensor.matmul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorMatMulWithTranspose) {
    auto a = MakeTensorVar("a", {8, 4});
    auto b = MakeTensorVar("b", {8, 16});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"a_trans",  std::any(true)},
        {"b_trans", std::any(false)}
    };
    auto call = registry_.Create("tensor.matmul", {a, b}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorMatMulVecVec) {
    auto a = MakeTensorVar("a", {8});
    auto b = MakeTensorVar("b", {8});
    auto call = registry_.Create("tensor.matmul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 0);
}

TEST_F(TensorOpsTest, TestTensorMatMulMatVec) {
    auto a = MakeTensorVar("a", {4, 8});
    auto b = MakeTensorVar("b", {8});
    auto call = registry_.Create("tensor.matmul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 1);
}

TEST_F(TensorOpsTest, TestTensorMatMulVecMat) {
    auto a = MakeTensorVar("a", {8});
    auto b = MakeTensorVar("b", {8, 16});
    auto call = registry_.Create("tensor.matmul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 1);
}

TEST_F(TensorOpsTest, TestTensorMatMulBatched) {
    auto a = MakeTensorVar("a", {2, 4, 8});
    auto b = MakeTensorVar("b", {2, 8, 16});
    auto call = registry_.Create("tensor.matmul", {a, b}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 3);
}

TEST_F(TensorOpsTest, TestTensorMatMulWithOutDtype) {
    auto a = MakeTensorVar("a", {4, 8}, DataType::FP16);
    auto b = MakeTensorVar("b", {8, 16}, DataType::FP16);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"out_dtype", std::any(DataType::FP32)}
    };
    auto call = registry_.Create("tensor.matmul", {a, b}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::FP32);
}

// ================================================================
// Memory Ops (tensor.create, tensor.view, tensor.assemble)
// ================================================================

TEST_F(TensorOpsTest, TestTensorCreateCreate) {
    auto shape = MakeShapeTuple({4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", std::any(DataType::FP32)}
    };
    auto call = registry_.Create("tensor.create", {shape}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
    ASSERT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(TensorOpsTest, TestTensorViewCreate) {
    auto input = MakeTensorVar("input", {16, 32});
    auto shape = MakeShapeTuple({4, 8});
    auto offset = MakeShapeTuple({0, 0});
    auto call = registry_.Create("tensor.view", {input, shape, offset}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorAssembleCreate) {
    auto target = MakeTensorVar("target", {16, 32});
    auto source = MakeTensorVar("source", {4, 8});
    auto offset = MakeShapeTuple({0, 0});
    auto call = registry_.Create("tensor.assemble", {target, source, offset}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

// ================================================================
// Reduction Ops (tensor.row_max, tensor.row_sum)
// ================================================================

TEST_F(TensorOpsTest, TestTensorRowMaxCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {    "axis",    std::any(1)},
        {"keep_dim", std::any(true)}
    };
    auto call = registry_.Create("tensor.row_max", {a}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(TensorOpsTest, TestTensorRowSumCreate) {
    auto a = MakeTensorVar("a", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {    "axis",     std::any(0)},
        {"keep_dim", std::any(false)}
    };
    auto call = registry_.Create("tensor.row_sum", {a}, kwargs, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(TensorOpsTest, TestTensorRowMaxNegativeAxis) {
    auto a = MakeTensorVar("a", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {    "axis",   std::any(-1)},
        {"keep_dim", std::any(true)}
    };
    auto call = registry_.Create("tensor.row_max", {a}, kwargs, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(TensorOpsTest, TestTensorRowMaxNoKeepDim) {
    auto a = MakeTensorVar("a", {4, 8});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {    "axis",     std::any(1)},
        {"keep_dim", std::any(false)}
    };
    auto call = registry_.Create("tensor.row_max", {a}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 1);
}

// ================================================================
// Transform Ops (tensor.reshape, tensor.transpose)
// ================================================================

TEST_F(TensorOpsTest, TestTensorReshapeCreate) {
    auto input = MakeTensorVar("input", {4, 8});
    auto shape = MakeShapeTuple({2, 16});
    auto call = registry_.Create("tensor.reshape", {input, shape}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorTransposeCreate) {
    auto input = MakeTensorVar("input", {4, 8});
    auto axis1 = std::make_shared<ConstInt>(0, DataType::INT32, span_);
    auto axis2 = std::make_shared<ConstInt>(1, DataType::INT32, span_);
    auto call = registry_.Create("tensor.transpose", {input, axis1, axis2}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorTranspose3D) {
    auto input = MakeTensorVar("input", {2, 4, 8});
    auto axis1 = std::make_shared<ConstInt>(0, DataType::INT32, span_);
    auto axis2 = std::make_shared<ConstInt>(2, DataType::INT32, span_);
    auto call = registry_.Create("tensor.transpose", {input, axis1, axis2}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 3);
}

// ================================================================
// OpRegistry Entry Metadata Tests
// ================================================================

TEST_F(TensorOpsTest, TestTensorAddEntryMetadata) {
    const auto &entry = registry_.GetEntry("tensor.add");
    ASSERT_EQ(entry.GetOpCategory(), "TensorOp");
    ASSERT_FALSE(entry.GetDescription().empty());
}

TEST_F(TensorOpsTest, TestTensorMatMulEntryMetadata) {
    const auto &entry = registry_.GetEntry("tensor.matmul");
    ASSERT_EQ(entry.GetOpCategory(), "TensorOp");
}

TEST_F(TensorOpsTest, TestTensorExpEntryMetadata) {
    const auto &entry = registry_.GetEntry("tensor.exp");
    ASSERT_EQ(entry.GetOpCategory(), "TensorOp");
}

TEST_F(TensorOpsTest, TestTensorCreateEntryMetadata) {
    const auto &entry = registry_.GetEntry("tensor.create");
    ASSERT_EQ(entry.GetOpCategory(), "TensorOp");
}

TEST_F(TensorOpsTest, TestIsRegistered) {
    ASSERT_TRUE(registry_.IsRegistered("tensor.add"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.sub"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.mul"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.div"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.maximum"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.add_scalar"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.sub_scalar"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.mul_scalar"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.div_scalar"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.exp"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.cast"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.matmul"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.create"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.view"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.assemble"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.row_max"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.row_sum"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.reshape"));
    ASSERT_TRUE(registry_.IsRegistered("tensor.transpose"));
}

// ================================================================
// Memory Ops - Runtime Tuple and Edge Case Tests
// ================================================================

TEST_F(TensorOpsTest, TestTensorCreateRuntimeShape) {
    // Test with runtime tuple (Var with TupleType) instead of MakeTuple
    auto tupleType = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64), std::make_shared<ScalarType>(DataType::INT64)});
    auto shapeVar = std::make_shared<Var>("shape", tupleType, span_);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", std::any(DataType::FP32)}
    };
    auto call = registry_.Create("tensor.create", {shapeVar}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorViewRuntimeShape) {
    auto input = MakeTensorVar("input", {16, 32});
    // Runtime shape tuple (Var with TupleType instead of MakeTuple)
    auto tupleType = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64), std::make_shared<ScalarType>(DataType::INT64)});
    auto shapeVar = std::make_shared<Var>("shape", tupleType, span_);
    auto offset = MakeShapeTuple({0, 0});
    auto call = registry_.Create("tensor.view", {input, shapeVar, offset}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorCreateUINT64Shape) {
    // Test with UINT64 shape elements
    std::vector<ExprPtr> elems;
    elems.push_back(std::make_shared<ConstInt>(4, DataType::UINT64, span_));
    elems.push_back(std::make_shared<ConstInt>(8, DataType::UINT64, span_));
    auto shape = std::make_shared<MakeTuple>(elems, span_);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", std::any(DataType::FP16)}
    };
    auto call = registry_.Create("tensor.create", {shape}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(TensorOpsTest, TestTensorViewUINT64Shape) {
    auto input = MakeTensorVar("input", {16, 32});
    // UINT64 shape elements
    std::vector<ExprPtr> shapeElems;
    shapeElems.push_back(std::make_shared<ConstInt>(4, DataType::UINT64, span_));
    shapeElems.push_back(std::make_shared<ConstInt>(8, DataType::UINT64, span_));
    auto shape = std::make_shared<MakeTuple>(shapeElems, span_);
    // UINT64 offset elements
    std::vector<ExprPtr> offsetElems;
    offsetElems.push_back(std::make_shared<ConstInt>(0, DataType::UINT64, span_));
    offsetElems.push_back(std::make_shared<ConstInt>(0, DataType::UINT64, span_));
    auto offset = std::make_shared<MakeTuple>(offsetElems, span_);
    auto call = registry_.Create("tensor.view", {input, shape, offset}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorAssemblePreservesTargetType) {
    auto target = MakeTensorVar("target", {16, 32}, DataType::FP16);
    auto source = MakeTensorVar("source", {4, 8}, DataType::FP16);
    auto offset = MakeShapeTuple({0, 0});
    auto call = registry_.Create("tensor.assemble", {target, source, offset}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->dtype_, DataType::FP16);
    ASSERT_EQ(rt->shape_.size(), 2);
}

TEST_F(TensorOpsTest, TestTensorAssembleUINT64Offset) {
    auto target = MakeTensorVar("target", {16, 32});
    auto source = MakeTensorVar("source", {4, 8});
    // UINT64 offset elements
    std::vector<ExprPtr> offsetElems;
    offsetElems.push_back(std::make_shared<ConstInt>(0, DataType::UINT64, span_));
    offsetElems.push_back(std::make_shared<ConstInt>(0, DataType::UINT64, span_));
    auto offset = std::make_shared<MakeTuple>(offsetElems, span_);
    auto call = registry_.Create("tensor.assemble", {target, source, offset}, span_);
    ASSERT_NE(call, nullptr);
}

TEST_F(TensorOpsTest, TestTensorCreate1D) {
    auto shape = MakeShapeTuple({64});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", std::any(DataType::INT32)}
    };
    auto call = registry_.Create("tensor.create", {shape}, kwargs, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 1);
    ASSERT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(TensorOpsTest, TestTensorViewChangeRank) {
    // View can change the rank of the tensor
    auto input = MakeTensorVar("input", {16, 32});
    auto shape = MakeShapeTuple({4, 4, 8});
    auto offset = MakeShapeTuple({0, 0, 0});
    auto call = registry_.Create("tensor.view", {input, shape, offset}, span_);
    ASSERT_NE(call, nullptr);
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->shape_.size(), 3);
    ASSERT_EQ(rt->dtype_, DataType::FP32);
}

} // namespace ir
} // namespace pypto
