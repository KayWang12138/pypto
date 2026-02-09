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
 * \file test_type.cpp
 * \brief Unit tests for IR type system
 */

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// ============================================================================
// Type Base Class Tests
// ============================================================================

TEST(IRTypeTest, TestTypeBasic) {
  // Test base Type class via UnknownType (Type is abstract with pure virtual GetKind)
  auto type = std::make_shared<UnknownType>();
  ASSERT_NE(type, nullptr);
  ASSERT_EQ(type->TypeName(), "UnknownType");
}

// ============================================================================
// UnknownType Tests
// ============================================================================

TEST(IRTypeTest, TestUnknownTypeConstructor) {
  // Test UnknownType construction
  auto unknown_type = std::make_shared<UnknownType>();
  ASSERT_NE(unknown_type, nullptr);
  ASSERT_EQ(unknown_type->TypeName(), "UnknownType");
}

TEST(IRTypeTest, TestGetUnknownTypeSingleton) {
  // Test GetUnknownType returns singleton
  auto type1 = GetUnknownType();
  auto type2 = GetUnknownType();
  ASSERT_EQ(type1, type2);  // Should be same instance
  ASSERT_EQ(type1->TypeName(), "UnknownType");
}

// ============================================================================
// ScalarType Tests
// ============================================================================

TEST(IRTypeTest, TestScalarTypeInt32) {
  // Test ScalarType with INT32
  auto scalar_type = std::make_shared<ScalarType>(DataType::INT32);
  ASSERT_NE(scalar_type, nullptr);
  ASSERT_EQ(scalar_type->TypeName(), "ScalarType");
  ASSERT_EQ(scalar_type->dtype_, DataType::INT32);
}

TEST(IRTypeTest, TestScalarTypeFloat32) {
  // Test ScalarType with FLOAT32
  auto scalar_type = std::make_shared<ScalarType>(DataType::FP32);
  ASSERT_NE(scalar_type, nullptr);
  ASSERT_EQ(scalar_type->dtype_, DataType::FP32);
}

TEST(IRTypeTest, TestScalarTypeBool) {
  // Test ScalarType with BOOL
  auto scalar_type = std::make_shared<ScalarType>(DataType::BOOL);
  ASSERT_NE(scalar_type, nullptr);
  ASSERT_EQ(scalar_type->dtype_, DataType::BOOL);
}

TEST(IRTypeTest, TestScalarTypeVariousDtypes) {
  // Test ScalarType with various data types
  std::vector<DataType> dtypes = {
      DataType::INT8,   DataType::INT16,  DataType::INT32,  DataType::INT64,
      DataType::UINT8,  DataType::UINT16, DataType::UINT32, DataType::UINT64,
      DataType::FP16, DataType::FP32, DataType::BOOL};

  for (const auto& dtype : dtypes) {
    auto scalar_type = std::make_shared<ScalarType>(dtype);
    ASSERT_NE(scalar_type, nullptr);
    ASSERT_EQ(scalar_type->dtype_, dtype);
  }
}

// ============================================================================
// TileView Tests
// ============================================================================

TEST(IRTypeTest, TestTileViewDefaultConstructor) {
  // Test TileView default construction
  TileView tile_view;
  ASSERT_TRUE(tile_view.valid_shape.empty());
  ASSERT_TRUE(tile_view.stride.empty());
  ASSERT_EQ(tile_view.start_offset, nullptr);
}

TEST(IRTypeTest, TestTileViewWithParameters) {
  // Test TileView with parameters
  auto const1 = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  auto const2 = std::make_shared<ConstInt>(32, DataType::INT32, Span::unknown());
  auto const3 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto offset = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());

  std::vector<ExprPtr> valid_shape = {const1, const2};
  std::vector<ExprPtr> stride = {const3, const3};

  TileView tile_view(valid_shape, stride, offset);
  ASSERT_EQ(tile_view.valid_shape.size(), 2);
  ASSERT_EQ(tile_view.stride.size(), 2);
  ASSERT_NE(tile_view.start_offset, nullptr);
}

// ============================================================================
// ShapedType Tests
// ============================================================================

TEST(IRTypeTest, TestShapedTypeWithoutMemRef) {
  // Test ShapedType without memory reference
  auto dim1 = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto dim2 = std::make_shared<ConstInt>(20, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim1, dim2};

  auto shaped_type = std::make_shared<ShapedType>(DataType::FP32, shape);
  ASSERT_NE(shaped_type, nullptr);
  ASSERT_EQ(shaped_type->TypeName(), "ShapedType");
  ASSERT_EQ(shaped_type->dtype_, DataType::FP32);
  ASSERT_EQ(shaped_type->shape_.size(), 2);
  ASSERT_FALSE(shaped_type->memref_.has_value());
}

TEST(IRTypeTest, TestShapedTypeWithMemRef) {
  // Test ShapedType with memory reference
  auto dim1 = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim1};

  auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::unknown());
  MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::UB, addr, 1024, 0);

  auto shaped_type = std::make_shared<ShapedType>(DataType::INT32, shape, memref);
  ASSERT_NE(shaped_type, nullptr);
  ASSERT_TRUE(shaped_type->memref_.has_value());
  ASSERT_EQ((*shaped_type->memref_)->memory_space_, MemorySpace::UB);
  ASSERT_EQ((*shaped_type->memref_)->size_, 1024);
}

// ============================================================================
// TensorType Tests
// ============================================================================

TEST(IRTypeTest, TestTensorTypeBasic) {
  // Test basic TensorType construction
  auto dim1 = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto dim2 = std::make_shared<ConstInt>(20, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim1, dim2};

  auto tensor_type = std::make_shared<TensorType>(shape, DataType::FP32);
  ASSERT_NE(tensor_type, nullptr);
  ASSERT_EQ(tensor_type->TypeName(), "TensorType");
  ASSERT_EQ(tensor_type->dtype_, DataType::FP32);
  ASSERT_EQ(tensor_type->shape_.size(), 2);
}

TEST(IRTypeTest, TestTensorType1D) {
  // Test 1D TensorType
  auto dim = std::make_shared<ConstInt>(100, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim};

  auto tensor_type = std::make_shared<TensorType>(shape, DataType::INT32);
  ASSERT_NE(tensor_type, nullptr);
  ASSERT_EQ(tensor_type->shape_.size(), 1);
}

TEST(IRTypeTest, TestTensorType3D) {
  // Test 3D TensorType
  auto dim1 = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto dim2 = std::make_shared<ConstInt>(20, DataType::INT32, Span::unknown());
  auto dim3 = std::make_shared<ConstInt>(30, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim1, dim2, dim3};

  auto tensor_type = std::make_shared<TensorType>(shape, DataType::FP16);
  ASSERT_NE(tensor_type, nullptr);
  ASSERT_EQ(tensor_type->shape_.size(), 3);
  ASSERT_EQ(tensor_type->dtype_, DataType::FP16);
}

TEST(IRTypeTest, TestTensorTypeWithMemRef) {
  // Test TensorType with memory reference
  auto dim = std::make_shared<ConstInt>(100, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim};

  auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::unknown());
  MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::DDR, addr, 400, 0);

  auto tensor_type = std::make_shared<TensorType>(shape, DataType::INT32, memref);
  ASSERT_NE(tensor_type, nullptr);
  ASSERT_TRUE(tensor_type->memref_.has_value());
  ASSERT_EQ((*tensor_type->memref_)->memory_space_, MemorySpace::DDR);
}

// ============================================================================
// TileType Tests
// ============================================================================

TEST(IRTypeTest, TestTileType1D) {
  // Test 1D TileType
  auto dim = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim};

  auto tile_type = std::make_shared<TileType>(shape, DataType::FP32);
  ASSERT_NE(tile_type, nullptr);
  ASSERT_EQ(tile_type->TypeName(), "TileType");
  ASSERT_EQ(tile_type->shape_.size(), 1);
  ASSERT_FALSE(tile_type->tile_view_.has_value());
}

TEST(IRTypeTest, TestTileType2D) {
  // Test 2D TileType (maximum allowed dimensions)
  auto dim1 = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  auto dim2 = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim1, dim2};

  auto tile_type = std::make_shared<TileType>(shape, DataType::FP16);
  ASSERT_NE(tile_type, nullptr);
  ASSERT_EQ(tile_type->shape_.size(), 2);
  ASSERT_EQ(tile_type->dtype_, DataType::FP16);
}

TEST(IRTypeTest, TestTileTypeWithMemRef) {
  // Test TileType with memory reference
  auto dim1 = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  auto dim2 = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim1, dim2};

  auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::unknown());
  MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::L0A, addr, 512, 0);

  auto tile_type = std::make_shared<TileType>(shape, DataType::FP32, memref);
  ASSERT_NE(tile_type, nullptr);
  ASSERT_TRUE(tile_type->memref_.has_value());
  ASSERT_EQ((*tile_type->memref_)->memory_space_, MemorySpace::L0A);
}

TEST(IRTypeTest, TestTileTypeWithTileView) {
  // Test TileType with tile view
  auto dim1 = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  auto dim2 = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim1, dim2};

  auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::unknown());
  MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::L0C, addr, 512, 0);

  auto valid_dim1 = std::make_shared<ConstInt>(8, DataType::INT32, Span::unknown());
  auto valid_dim2 = std::make_shared<ConstInt>(8, DataType::INT32, Span::unknown());
  auto stride1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto stride2 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto offset = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());

  TileView tile_view({valid_dim1, valid_dim2}, {stride1, stride2}, offset);

  auto tile_type = std::make_shared<TileType>(shape, DataType::FP32, memref, tile_view);
  ASSERT_NE(tile_type, nullptr);
  ASSERT_TRUE(tile_type->tile_view_.has_value());
  ASSERT_EQ(tile_type->tile_view_->valid_shape.size(), 2);
  ASSERT_EQ(tile_type->tile_view_->stride.size(), 2);
}

TEST(IRTypeTest, TestTileTypeInvalidDimensions) {
  // Test that TileType with more than 2 dimensions can be created
  // (dimension limit is now enforced at code generation level, not type level)
  auto dim1 = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  auto dim2 = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  auto dim3 = std::make_shared<ConstInt>(16, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim1, dim2, dim3};

  auto tile_type = std::make_shared<TileType>(shape, DataType::FP32);
  ASSERT_NE(tile_type, nullptr);
  ASSERT_EQ(tile_type->shape_.size(), 3);
}

// ============================================================================
// TupleType Tests
// ============================================================================

TEST(IRTypeTest, TestTupleTypeEmpty) {
  // Test empty TupleType
  std::vector<TypePtr> types;
  auto tuple_type = std::make_shared<TupleType>(types);
  ASSERT_NE(tuple_type, nullptr);
  ASSERT_EQ(tuple_type->TypeName(), "TupleType");
  ASSERT_TRUE(tuple_type->types_.empty());
}

TEST(IRTypeTest, TestTupleTypeSingleElement) {
  // Test TupleType with single element
  auto scalar_type = std::make_shared<ScalarType>(DataType::INT32);
  std::vector<TypePtr> types = {scalar_type};

  auto tuple_type = std::make_shared<TupleType>(types);
  ASSERT_NE(tuple_type, nullptr);
  ASSERT_EQ(tuple_type->types_.size(), 1);
}

TEST(IRTypeTest, TestTupleTypeMultipleElements) {
  // Test TupleType with multiple elements
  auto scalar_type = std::make_shared<ScalarType>(DataType::INT32);
  auto dim = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim};
  auto tensor_type = std::make_shared<TensorType>(shape, DataType::FP32);

  std::vector<TypePtr> types = {scalar_type, tensor_type};
  auto tuple_type = std::make_shared<TupleType>(types);
  ASSERT_NE(tuple_type, nullptr);
  ASSERT_EQ(tuple_type->types_.size(), 2);
}

TEST(IRTypeTest, TestTupleTypeNested) {
  // Test nested TupleType
  auto scalar_type1 = std::make_shared<ScalarType>(DataType::INT32);
  auto scalar_type2 = std::make_shared<ScalarType>(DataType::FP32);

  std::vector<TypePtr> inner_types = {scalar_type1, scalar_type2};
  auto inner_tuple = std::make_shared<TupleType>(inner_types);

  std::vector<TypePtr> outer_types = {inner_tuple, scalar_type1};
  auto outer_tuple = std::make_shared<TupleType>(outer_types);

  ASSERT_NE(outer_tuple, nullptr);
  ASSERT_EQ(outer_tuple->types_.size(), 2);
}

TEST(IRTypeTest, TestTupleTypeWithTensorAndScalar) {
  // Test TupleType with mixed tensor and scalar types
  auto scalar_type = std::make_shared<ScalarType>(DataType::BOOL);

  auto dim1 = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto dim2 = std::make_shared<ConstInt>(20, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> shape = {dim1, dim2};
  auto tensor_type = std::make_shared<TensorType>(shape, DataType::FP32);

  std::vector<TypePtr> types = {tensor_type, scalar_type};
  auto tuple_type = std::make_shared<TupleType>(types);

  ASSERT_NE(tuple_type, nullptr);
  ASSERT_EQ(tuple_type->types_.size(), 2);
  ASSERT_EQ(tuple_type->types_[0]->TypeName(), "TensorType");
  ASSERT_EQ(tuple_type->types_[1]->TypeName(), "ScalarType");
}

// ============================================================================
// Type Polymorphism Tests
// ============================================================================

TEST(IRTypeTest, TestTypePolymorphism) {
  // Test that derived types can be used as base Type pointers
  TypePtr type1 = std::make_shared<UnknownType>();
  TypePtr type2 = std::make_shared<ScalarType>(DataType::INT32);
  TypePtr type3 = std::make_shared<TupleType>(std::vector<TypePtr>{});

  ASSERT_EQ(type1->TypeName(), "UnknownType");
  ASSERT_EQ(type2->TypeName(), "ScalarType");
  ASSERT_EQ(type3->TypeName(), "TupleType");
}

TEST(IRTypeTest, TestTypeDynamicCast) {
  // Test dynamic casting of types
  TypePtr base_type = std::make_shared<ScalarType>(DataType::FP32);

  auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(base_type);
  ASSERT_NE(scalar_type, nullptr);
  ASSERT_EQ(scalar_type->dtype_, DataType::FP32);

  auto tensor_type = std::dynamic_pointer_cast<const TensorType>(base_type);
  ASSERT_EQ(tensor_type, nullptr);  // Should fail
}

}  // namespace ir
}  // namespace pypto
