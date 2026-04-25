/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file backend_910b_cce_ops.cpp
 * @brief Backend op registration for Backend910B_CCE
 *
 * This file registers all block operations for the CCE backend.
 * Each registration specifies the pipe type and CCE codegen function.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "block/backend/910B_CCE/backend_910b_cce.h"
#include "block/backend/common/backend.h"
#include "block/codegen/cce/cce_codegen.h"
#include "block/codegen/codegen_base.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/pipe.h"
#include "ir/type.h"

namespace pypto {
namespace backend {
using ir::DataType;

// ============================================================================
// Helper Functions for CCE Code Generation
// ============================================================================

/**
 * @brief Compute stride-based offset for multi-dimensional tensor access
 * @param codegen The CCE codegen instance
 * @param tensor_var_name The tensor variable name (e.g., "inputGlobal")
 * @param offset_exprs Vector of offset expressions for each dimension
 * @param tensor_type The tensor type for shape information
 * @return C++ expression string for total offset computation
 */
static std::string ComputeStrideBasedOffset(codegen::CCECodegen& codegen, const std::string& tensor_var_name,
                                            const ir::MakeTuplePtr& offsets,
                                            const ir::TensorTypePtr& tensor_type) {
  // In single-file mode: compute strides from IR tensor shape (no Tensor struct)
  if (codegen.IsSingleFileMode()) {
    return codegen.ComputeIRBasedOffset(tensor_type, offsets);
  }

  // Get Tensor struct pointer for stride access
  std::string tensor_struct = codegen.GetTensorStruct(tensor_var_name);

  // Build offset computation: offset[0] * stride[0] + offset[1] * stride[1] + ...
  std::ostringstream offset_computation;
  offset_computation << "(" << tensor_struct << "->start_offset";

  for (size_t i = 0; i < offsets->elements_.size(); ++i) {
    offset_computation << " + " << codegen.GetExprAsCode(offsets->elements_[i]) << " * " << tensor_struct
                       << "->strides[" << i << "]";
  }

  offset_computation << ")";
  return offset_computation.str();
}

static bool IsNZTensorType(const ir::TensorTypePtr& tensor_type) {
  return tensor_type && tensor_type->tensor_view_.has_value() &&
         tensor_type->tensor_view_->layout == ir::TensorLayout::NZ;
}

static int64_t GetNZInnerCols(const DataType& dtype) {
  if (dtype == DataType::BOOL || dtype == DataType::INT8 || dtype == DataType::UINT8) {
    return 32;
  }
  if (dtype == DataType::FP16 || dtype == DataType::BF16 ||
      dtype == DataType::INT16 || dtype == DataType::UINT16) {
    return 16;
  }
  if (dtype == DataType::FP32 || dtype == DataType::INT32 || dtype == DataType::UINT32) {
    return 8;
  }
  if (dtype == DataType::INT64 || dtype == DataType::UINT64) {
    return 4;
  }
  throw pypto::ir::ValueError("CCE NZ store does not support destination dtype " + dtype.ToString());
}

static int64_t GetStaticConstIntOrThrow(const ir::ExprPtr& expr, const std::string& message) {
  auto value = ir::As<ir::ConstInt>(expr);
  if (!value) {
    throw pypto::ir::ValueError(message);
  }
  return value->value_;
}

static bool IsConstZero(const ir::ExprPtr& expr) {
  auto value = ir::As<ir::ConstInt>(expr);
  return value && value->value_ == 0;
}

static void ValidateCCEStoreNZPreconditions(const std::string& op_name,
                                            const ir::ExprPtr& src_expr,
                                            const ir::MakeTuplePtr& offsets,
                                            const ir::TensorTypePtr& dst_tensor_type) {
  if (!IsNZTensorType(dst_tensor_type)) {
    return;
  }

  auto src_tile_type = ir::As<ir::TileType>(src_expr->GetType());
  CHECK(src_tile_type != nullptr) << op_name << ": source must be TileType";
  CHECK(src_tile_type->memref_.has_value()) << op_name << ": source tile must have an allocated memory space";
  if (src_tile_type->memref_.value()->memory_space_ != ir::MemorySpace::Acc) {
    throw pypto::ir::ValueError(op_name + ": CCE NZ output currently only supports Acc source tiles");
  }

  if (dst_tensor_type->shape_.size() != 2) {
    throw pypto::ir::ValueError(op_name + ": CCE NZ output currently requires a 2D destination tensor");
  }
  if (offsets->elements_.size() != 2 || !IsConstZero(offsets->elements_[0]) || !IsConstZero(offsets->elements_[1])) {
    throw pypto::ir::ValueError(op_name + ": CCE NZ output currently requires offsets=[0, 0]");
  }

  const int64_t rows = GetStaticConstIntOrThrow(
      dst_tensor_type->shape_[0], op_name + ": CCE NZ output currently requires a static destination row shape");
  const int64_t cols = GetStaticConstIntOrThrow(
      dst_tensor_type->shape_[1], op_name + ": CCE NZ output currently requires a static destination column shape");
  const int64_t c0 = GetNZInnerCols(dst_tensor_type->dtype_);
  if (rows % 16 != 0 || cols % c0 != 0) {
    throw pypto::ir::ValueError(
        op_name + ": CCE NZ output requires rows divisible by 16 and cols divisible by the destination C0 size");
  }
}

static std::string BuildStaticNZShapeType(const DataType& dtype, int64_t rows, int64_t cols) {
  const int64_t c0 = GetNZInnerCols(dtype);
  return "pto::Shape<1, " + std::to_string(cols / c0) + ", " + std::to_string(rows / 16) + ", 16, " +
         std::to_string(c0) + ">";
}

static std::string BuildStaticNZStrideType(const DataType& dtype, int64_t full_rows, int64_t full_cols) {
  const int64_t c0 = GetNZInnerCols(dtype);
  return "pto::Stride<" + std::to_string(full_rows * full_cols) + ", " + std::to_string(full_rows * c0) + ", " +
         std::to_string(16 * c0) + ", " + std::to_string(c0) + ", 1>";
}

static int64_t ComputeStaticNZPhysicalOffset(const DataType& dtype, int64_t full_rows, int64_t row_offset,
                                             int64_t col_offset) {
  const int64_t c0 = GetNZInnerCols(dtype);
  return col_offset * full_rows + row_offset * c0;
}

static void ValidateDebugDumpNZWindowPreconditions(const ir::TensorTypePtr& tensor_type,
                                                   const ir::MakeTuplePtr& offsets,
                                                   const ir::MakeTuplePtr& shapes) {
  CHECK(tensor_type != nullptr) << "debug.dump_tensor NZ lowering requires TensorType";
  CHECK(offsets != nullptr) << "debug.dump_tensor NZ lowering requires offsets tuple";
  CHECK(shapes != nullptr) << "debug.dump_tensor NZ lowering requires shapes tuple";

  if (tensor_type->shape_.size() != 2 || offsets->elements_.size() != 2 || shapes->elements_.size() != 2) {
    throw pypto::ir::ValueError("debug.dump_tensor: CCE NZ dump currently requires a 2D tensor and 2D offsets/shapes");
  }

  const int64_t full_rows = GetStaticConstIntOrThrow(
      tensor_type->shape_[0], "debug.dump_tensor: CCE NZ dump currently requires a static destination row shape");
  const int64_t full_cols = GetStaticConstIntOrThrow(
      tensor_type->shape_[1], "debug.dump_tensor: CCE NZ dump currently requires a static destination column shape");
  const int64_t row_offset = GetStaticConstIntOrThrow(
      offsets->elements_[0], "debug.dump_tensor: CCE NZ dump currently only supports aligned static windows");
  const int64_t col_offset = GetStaticConstIntOrThrow(
      offsets->elements_[1], "debug.dump_tensor: CCE NZ dump currently only supports aligned static windows");
  const int64_t rows = GetStaticConstIntOrThrow(
      shapes->elements_[0], "debug.dump_tensor: CCE NZ dump currently only supports aligned static windows");
  const int64_t cols = GetStaticConstIntOrThrow(
      shapes->elements_[1], "debug.dump_tensor: CCE NZ dump currently only supports aligned static windows");
  const int64_t c0 = GetNZInnerCols(tensor_type->dtype_);

  if (full_rows % 16 != 0 || full_cols % c0 != 0) {
    throw pypto::ir::ValueError(
        "debug.dump_tensor: CCE NZ dump requires rows divisible by 16 and cols divisible by the destination C0 size");
  }
  if (row_offset < 0 || col_offset < 0 || rows <= 0 || cols <= 0) {
    throw pypto::ir::ValueError("debug.dump_tensor: CCE NZ dump requires non-negative offsets and positive shapes");
  }
  if (row_offset % 16 != 0 || rows % 16 != 0 || col_offset % c0 != 0 || cols % c0 != 0) {
    throw pypto::ir::ValueError("debug.dump_tensor: CCE NZ dump currently only supports aligned static windows");
  }
  if (row_offset + rows > full_rows || col_offset + cols > full_cols) {
    throw pypto::ir::ValueError("debug.dump_tensor: CCE NZ dump window must stay within tensor bounds");
  }
}

static int NextDebugDumpId() {
  static int next_debug_dump_id = 0;
  return next_debug_dump_id++;
}

static std::string JoinExpressions(const std::vector<std::string>& expressions, const std::string& delimiter) {
  std::ostringstream oss;
  for (size_t i = 0; i < expressions.size(); ++i) {
    if (i > 0) oss << delimiter;
    oss << expressions[i];
  }
  return oss.str();
}

static std::string EscapeCxxStringLiteral(const std::string& text) {
  std::ostringstream oss;
  for (char c : text) {
    switch (c) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\t':
        oss << "\\t";
        break;
      case '\r':
        oss << "\\r";
        break;
      default:
        oss << c;
        break;
    }
  }
  return oss.str();
}

static std::string FormatDebugLocation(const ir::Span& span) {
  if (!span.is_valid() || span.filename_.empty() || span.begin_line_ <= 0) {
    return "";
  }

  size_t last_sep = span.filename_.find_last_of("/\\");
  std::string basename = last_sep == std::string::npos ? span.filename_ : span.filename_.substr(last_sep + 1);
  if (basename.empty()) {
    return "";
  }
  return "[" + basename + ":" + std::to_string(span.begin_line_) + "]";
}

static std::string FormatDebugLocationHeader(const ir::Span& span, const std::string& op_name) {
  std::string location = FormatDebugLocation(span);
  if (location.empty()) {
    return "";
  }
  return location + " " + op_name;
}

static void EmitDebugLocationHeaderCCE(codegen::CCECodegen& codegen, const ir::Span& span,
                                       const std::string& op_name) {
  std::string header = FormatDebugLocationHeader(span, op_name);
  if (!header.empty()) {
    codegen.Emit("cce::printf(\"" + EscapeCxxStringLiteral(header + "\n") + "\");");
  }
}

static bool HasDynamicTensorShape(const ir::TensorTypePtr& tensor_type) {
  for (const auto& dim : tensor_type->shape_) {
    if (!ir::As<ir::ConstInt>(dim)) {
      return true;
    }
  }
  return false;
}

static bool IsFullTensorWindow(const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets,
                               const ir::MakeTuplePtr& shapes) {
  if (!tensor_type || !offsets || !shapes) {
    return false;
  }
  const size_t rank = tensor_type->shape_.size();
  if (offsets->elements_.size() != rank || shapes->elements_.size() != rank) {
    return false;
  }

  for (size_t i = 0; i < rank; ++i) {
    auto offset_const = ir::As<ir::ConstInt>(offsets->elements_[i]);
    if (!offset_const || offset_const->value_ != 0) {
      return false;
    }

    auto shape_const = ir::As<ir::ConstInt>(shapes->elements_[i]);
    auto tensor_dim_const = ir::As<ir::ConstInt>(tensor_type->shape_[i]);
    if (shape_const && tensor_dim_const) {
      if (shape_const->value_ != tensor_dim_const->value_) {
        return false;
      }
      continue;
    }
    if (shapes->elements_[i].get() != tensor_type->shape_[i].get()) {
      return false;
    }
  }

  return true;
}

static std::string GetRuntimeTensorShapeExpr(const std::string& tensor_name, size_t rank, size_t axis) {
  const size_t gt_dim = 5 - rank + axis;
  return tensor_name + ".GetShape(GlobalTensorDim::DIM_" + std::to_string(gt_dim) + ")";
}

static std::string GetRuntimeTensorStrideExpr(const std::string& tensor_name, size_t rank, size_t axis) {
  const size_t gt_dim = 5 - rank + axis;
  return tensor_name + ".GetStride(GlobalTensorDim::DIM_" + std::to_string(gt_dim) + ")";
}

static std::string BuildShapeTypeForDump(codegen::CCECodegen& codegen, const std::string& tensor_name,
                                         const ir::TensorTypePtr& tensor_type,
                                         const std::vector<ir::ExprPtr>& shape_exprs, bool use_runtime_full_shape,
                                         std::vector<std::string>* ctor_args) {
  CHECK(shape_exprs.size() >= 1 && shape_exprs.size() <= 5)
      << "debug.dump_tensor currently supports tensor rank 1..5, but got " << shape_exprs.size();

  const size_t pad_dims = 5 - shape_exprs.size();
  std::vector<std::string> template_dims(5, "1");
  ctor_args->clear();
  for (size_t i = 0; i < shape_exprs.size(); ++i) {
    if (auto dim = ir::As<ir::ConstInt>(shape_exprs[i])) {
      template_dims[pad_dims + i] = std::to_string(dim->value_);
    } else {
      template_dims[pad_dims + i] = "-1";
      if (use_runtime_full_shape) {
        ctor_args->push_back(GetRuntimeTensorShapeExpr(tensor_name, tensor_type->shape_.size(), i));
      } else {
        ctor_args->push_back(codegen.GetExprAsCode(shape_exprs[i]));
      }
    }
  }
  return "pto::Shape<" + JoinExpressions(template_dims, ", ") + ">";
}

static std::string BuildStrideTypeForDump(codegen::CCECodegen& codegen, const std::string& tensor_name,
                                          const ir::TensorTypePtr& tensor_type, bool use_runtime_tensor_view,
                                          std::vector<std::string>* ctor_args) {
  CHECK(tensor_type) << "debug.dump_tensor requires TensorType for stride generation";
  const size_t rank = tensor_type->shape_.size();
  CHECK(rank >= 1 && rank <= 5) << "debug.dump_tensor currently supports tensor rank 1..5, but got " << rank;

  std::vector<std::string> stride_template_dims(5, "1");
  ctor_args->clear();
  const size_t pad_dims = 5 - rank;

  auto append_dynamic_stride = [&](size_t axis, const std::string& expr) {
    stride_template_dims[pad_dims + axis] = "-1";
    ctor_args->push_back(expr);
  };

  if (use_runtime_tensor_view) {
    for (size_t i = 0; i < rank; ++i) {
      append_dynamic_stride(i, GetRuntimeTensorStrideExpr(tensor_name, rank, i));
    }
    return "pto::Stride<" + JoinExpressions(stride_template_dims, ", ") + ">";
  }

  if (tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty()) {
    const auto& strides = tensor_type->tensor_view_->stride;
    CHECK(strides.size() == rank)
        << "debug.dump_tensor tensor_view stride rank (" << strides.size()
        << ") must match tensor rank (" << rank << ")";
    for (size_t i = 0; i < rank; ++i) {
      if (auto stride = ir::As<ir::ConstInt>(strides[i])) {
        stride_template_dims[pad_dims + i] = std::to_string(stride->value_);
      } else {
        append_dynamic_stride(i, codegen.GetExprAsCode(strides[i]));
      }
    }
    return "pto::Stride<" + JoinExpressions(stride_template_dims, ", ") + ">";
  }

  for (size_t i = 0; i < rank; ++i) {
    bool all_const = true;
    int64_t const_stride = 1;
    std::vector<std::string> factors;
    for (size_t j = i + 1; j < rank; ++j) {
      if (auto dim = ir::As<ir::ConstInt>(tensor_type->shape_[j])) {
        const_stride *= dim->value_;
      } else {
        all_const = false;
        factors.push_back(codegen.GetExprAsCode(tensor_type->shape_[j]));
      }
    }
    if (all_const) {
      stride_template_dims[pad_dims + i] = std::to_string(const_stride);
    } else {
      std::string expr = std::to_string(const_stride);
      if (!factors.empty()) {
        expr += " * " + JoinExpressions(factors, " * ");
      }
      append_dynamic_stride(i, "(" + expr + ")");
    }
  }

  return "pto::Stride<" + JoinExpressions(stride_template_dims, ", ") + ">";
}

static std::string ComputeRuntimeStrideBasedOffset(codegen::CCECodegen& codegen, const std::string& tensor_name,
                                                   const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets,
                                                   const std::string& start_offset) {
  CHECK(tensor_type) << "debug.dump_tensor requires TensorType for runtime offset generation";
  const size_t rank = tensor_type->shape_.size();
  CHECK(offsets) << "debug.dump_tensor requires offsets tuple for runtime offset generation";
  CHECK(offsets->elements_.size() == rank)
      << "debug.dump_tensor offset rank (" << offsets->elements_.size() << ") must match tensor rank (" << rank << ")";

  std::ostringstream offset_computation;
  offset_computation << "(";
  bool has_term = false;
  if (!start_offset.empty()) {
    offset_computation << start_offset;
    has_term = true;
  }

  for (size_t i = 0; i < rank; ++i) {
    if (has_term) {
      offset_computation << " + ";
    }
    offset_computation << codegen.GetExprAsCode(offsets->elements_[i]) << " * ";
    offset_computation << GetRuntimeTensorStrideExpr(tensor_name, rank, i);
    has_term = true;
  }

  if (!has_term) {
    offset_computation << "0";
  }
  offset_computation << ")";
  return offset_computation.str();
}

static std::string MakeDebugDumpTensorCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "debug.dump_tensor requires 3 arguments, but got " << op->args_.size();
  if (op->GetKwarg<bool>("show_location", false)) {
    EmitDebugLocationHeaderCCE(codegen, op->span_, "dump_tensor");
  }

  auto tensor_var = ir::As<ir::Var>(op->args_[0]);
  CHECK(tensor_var) << "debug.dump_tensor first argument must be a Var";
  auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
  CHECK(tensor_type) << "debug.dump_tensor first argument must be TensorType";
  auto offsets_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple) << "debug.dump_tensor second argument must be a tuple (offsets)";
  auto shapes_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple) << "debug.dump_tensor third argument must be a tuple (shapes)";

  if (IsNZTensorType(tensor_type)) {
    ValidateDebugDumpNZWindowPreconditions(tensor_type, offsets_tuple, shapes_tuple);

    const int debug_id = NextDebugDumpId();
    const std::string tensor_name = codegen.GetVarName(tensor_var);
    std::string base_ptr = codegen.GetPointer(tensor_name);
    if (base_ptr.empty()) {
      base_ptr = tensor_name + ".data()";
    }

    const int64_t full_rows = GetStaticConstIntOrThrow(
        tensor_type->shape_[0], "debug.dump_tensor: CCE NZ dump currently requires a static destination row shape");
    const int64_t full_cols = GetStaticConstIntOrThrow(
        tensor_type->shape_[1], "debug.dump_tensor: CCE NZ dump currently requires a static destination column shape");
    const int64_t row_offset = GetStaticConstIntOrThrow(
        offsets_tuple->elements_[0], "debug.dump_tensor: CCE NZ dump currently only supports aligned static windows");
    const int64_t col_offset = GetStaticConstIntOrThrow(
        offsets_tuple->elements_[1], "debug.dump_tensor: CCE NZ dump currently only supports aligned static windows");
    const int64_t rows = GetStaticConstIntOrThrow(
        shapes_tuple->elements_[0], "debug.dump_tensor: CCE NZ dump currently only supports aligned static windows");
    const int64_t cols = GetStaticConstIntOrThrow(
        shapes_tuple->elements_[1], "debug.dump_tensor: CCE NZ dump currently only supports aligned static windows");
    const int64_t physical_offset =
        ComputeStaticNZPhysicalOffset(tensor_type->dtype_, full_rows, row_offset, col_offset);
    std::string start_offset;
    if (!codegen.IsSingleFileMode()) {
      std::string tensor_struct = codegen.GetTensorStruct(tensor_name);
      if (!tensor_struct.empty()) {
        start_offset = tensor_struct + "->start_offset";
      }
    }
    std::string effective_offset = std::to_string(physical_offset);
    if (!start_offset.empty()) {
      effective_offset = "(" + start_offset + " + " + effective_offset + ")";
    }

    const std::string shape_alias = "__debug_dump_tensor_shape_" + std::to_string(debug_id);
    const std::string stride_alias = "__debug_dump_tensor_stride_" + std::to_string(debug_id);
    const std::string global_alias = "__debug_dump_tensor_type_" + std::to_string(debug_id);
    const std::string view_name = "__debug_dump_tensor_view_" + std::to_string(debug_id);

    codegen.Emit("using " + shape_alias + " = " + BuildStaticNZShapeType(tensor_type->dtype_, rows, cols) + ";");
    codegen.Emit("using " + stride_alias + " = " + BuildStaticNZStrideType(tensor_type->dtype_, full_rows, full_cols) +
                 ";");
    codegen.Emit("using " + global_alias + " = GlobalTensor<" + codegen.GetTypeString(tensor_type->dtype_) + ", " +
                 shape_alias + ", " + stride_alias + ", Layout::NZ>;");
    codegen.Emit(global_alias + " " + view_name + "(" + base_ptr + " + " + effective_offset + ");");
    codegen.Emit("TPRINT(" + view_name + ");");
    return "";
  }

  const int debug_id = NextDebugDumpId();
  const std::string tensor_name = codegen.GetVarName(tensor_var);
  std::string base_ptr = codegen.GetPointer(tensor_name);
  if (base_ptr.empty()) {
    base_ptr = tensor_name + ".data()";
  }
  const std::string shape_alias = "__debug_dump_tensor_shape_" + std::to_string(debug_id);
  const std::string stride_alias = "__debug_dump_tensor_stride_" + std::to_string(debug_id);
  const std::string global_alias = "__debug_dump_tensor_type_" + std::to_string(debug_id);
  const std::string view_name = "__debug_dump_tensor_view_" + std::to_string(debug_id);
  const bool has_dynamic_tensor_shape = HasDynamicTensorShape(tensor_type);
  const bool is_full_tensor_window = IsFullTensorWindow(tensor_type, offsets_tuple, shapes_tuple);
  const bool use_runtime_tensor_view = has_dynamic_tensor_shape;

  std::string start_offset;
  if (!codegen.IsSingleFileMode()) {
    std::string tensor_struct = codegen.GetTensorStruct(tensor_name);
    if (!tensor_struct.empty()) {
      start_offset = tensor_struct + "->start_offset";
    }
  }
  const std::string offset_expr =
      use_runtime_tensor_view
          ? ComputeRuntimeStrideBasedOffset(codegen, tensor_name, tensor_type, offsets_tuple, start_offset)
          : ComputeStrideBasedOffset(codegen, tensor_name, offsets_tuple, tensor_type);

  std::vector<std::string> shape_ctor_args;
  std::vector<std::string> stride_ctor_args;
  const std::string shape_type = BuildShapeTypeForDump(codegen, tensor_name, tensor_type, shapes_tuple->elements_,
                                                       use_runtime_tensor_view && is_full_tensor_window,
                                                       &shape_ctor_args);
  const std::string stride_type =
      BuildStrideTypeForDump(codegen, tensor_name, tensor_type, use_runtime_tensor_view, &stride_ctor_args);

  std::string layout_suffix = ", Layout::ND";
  if (tensor_type->shape_.size() == 2) {
    if (auto last_dim = ir::As<ir::ConstInt>(tensor_type->shape_.back())) {
      if (last_dim->value_ == 1) {
        layout_suffix = ", Layout::DN";
      }
    }
  }

  codegen.Emit("using " + shape_alias + " = " + shape_type + ";");
  codegen.Emit("using " + stride_alias + " = " + stride_type + ";");
  codegen.Emit("using " + global_alias + " = GlobalTensor<" + codegen.GetTypeString(tensor_type->dtype_) + ", " +
               shape_alias + ", " + stride_alias + layout_suffix + ">;");

  std::string shape_ctor = shape_alias + "(" + JoinExpressions(shape_ctor_args, ", ") + ")";
  if (shape_ctor_args.empty()) {
    shape_ctor = shape_alias + "()";
  }
  std::string stride_ctor = stride_alias + "(" + JoinExpressions(stride_ctor_args, ", ") + ")";
  if (stride_ctor_args.empty()) {
    stride_ctor = stride_alias + "()";
  }

  codegen.Emit(global_alias + " " + view_name + "(" + base_ptr + " + " + offset_expr + ", " + shape_ctor + ", " +
               stride_ctor + ");");
  codegen.Emit("TPRINT(" + view_name + ");");
  return "";
}

static std::string MakeDebugDumpTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1 || op->args_.size() == 3)
      << "debug.dump_tile requires 1 argument (tile) or 3 arguments (tile, offsets, shapes), but got "
      << op->args_.size();
  if (op->GetKwarg<bool>("show_location", false)) {
    EmitDebugLocationHeaderCCE(codegen, op->span_, "dump_tile");
  }

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  if (op->args_.size() == 1) {
    codegen.Emit("TPRINT(" + src + ");");
    return "";
  }

  auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  CHECK(tile_type) << "debug.dump_tile first argument must be TileType";
  CHECK(tile_type->shape_.size() == 2) << "debug.dump_tile CCE lowering currently only supports 2D tiles";
  auto offsets_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple) << "debug.dump_tile second argument must be a tuple (offsets)";
  auto shapes_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple) << "debug.dump_tile third argument must be a tuple (shapes)";

  auto tile_rows = ir::As<ir::ConstInt>(tile_type->shape_[0]);
  auto tile_cols = ir::As<ir::ConstInt>(tile_type->shape_[1]);
  CHECK(tile_rows && tile_cols) << "debug.dump_tile CCE lowering requires static physical tile shape";

  const int debug_id = NextDebugDumpId();
  const std::string requested_row = "__debug_dump_tile_requested_row_" + std::to_string(debug_id);
  const std::string requested_col = "__debug_dump_tile_requested_col_" + std::to_string(debug_id);
  const std::string src_valid_row = "__debug_dump_tile_src_valid_row_" + std::to_string(debug_id);
  const std::string src_valid_col = "__debug_dump_tile_src_valid_col_" + std::to_string(debug_id);
  const std::string valid_row = "__debug_dump_tile_valid_row_" + std::to_string(debug_id);
  const std::string valid_col = "__debug_dump_tile_valid_col_" + std::to_string(debug_id);
  const std::string row_idx = "__debug_dump_tile_r_" + std::to_string(debug_id);
  const std::string col_idx = "__debug_dump_tile_c_" + std::to_string(debug_id);
  const std::string row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  const std::string col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
  const std::string row_shape = codegen.GetExprAsCode(shapes_tuple->elements_[0]);
  const std::string col_shape = codegen.GetExprAsCode(shapes_tuple->elements_[1]);
  const std::string debug_val = "__debug_dump_tile_val_" + std::to_string(debug_id);

  codegen.Emit("pipe_barrier(PIPE_ALL);");
  codegen.Emit("int " + requested_row + " = " + row_shape + ";");
  codegen.Emit("if (" + requested_row + " < 0) " + requested_row + " = 0;");
  codegen.Emit("int " + requested_col + " = " + col_shape + ";");
  codegen.Emit("if (" + requested_col + " < 0) " + requested_col + " = 0;");
  codegen.Emit("int " + src_valid_row + " = " + src + ".GetValidRow() - (" + row_off + ");");
  codegen.Emit("if (" + src_valid_row + " < 0) " + src_valid_row + " = 0;");
  codegen.Emit("int " + src_valid_col + " = " + src + ".GetValidCol() - (" + col_off + ");");
  codegen.Emit("if (" + src_valid_col + " < 0) " + src_valid_col + " = 0;");
  codegen.Emit("int " + valid_row + " = " + requested_row + ";");
  codegen.Emit("if (" + valid_row + " > " + src_valid_row + ") " + valid_row + " = " + src_valid_row + ";");
  codegen.Emit("if (" + valid_row + " < 0) " + valid_row + " = 0;");
  codegen.Emit("if (" + valid_row + " > " + std::to_string(tile_rows->value_) + ") " + valid_row + " = " +
               std::to_string(tile_rows->value_) + ";");
  codegen.Emit("int " + valid_col + " = " + requested_col + ";");
  codegen.Emit("if (" + valid_col + " > " + src_valid_col + ") " + valid_col + " = " + src_valid_col + ";");
  codegen.Emit("if (" + valid_col + " < 0) " + valid_col + " = 0;");
  codegen.Emit("if (" + valid_col + " > " + std::to_string(tile_cols->value_) + ") " + valid_col + " = " +
               std::to_string(tile_cols->value_) + ";");
  codegen.Emit("cce::printf(\"=== [TPRINT Tile Window] Data Type: %s, Layout: %s, TileType: %s ===\\n\", "
               "pto::GetDTypeName<" + codegen.GetTypeString(tile_type->dtype_) +
               ">(), pto::GetLayoutName(decltype(" + src + ")::BFractal, decltype(" + src +
               ")::SFractal), \"Vec\");");
  codegen.Emit("cce::printf(\"  Source Shape: [%d, %d], Window Offsets: [%d, %d], Requested Shape: [%d, %d], "
               "Valid Shape: [%d, %d]\\n\", " + std::to_string(tile_rows->value_) + ", " +
               std::to_string(tile_cols->value_) + ", static_cast<int>(" + row_off + "), static_cast<int>(" + col_off +
               "), " + requested_row + ", " + requested_col + ", " + valid_row + ", " + valid_col + ");");
  codegen.Emit("for (int " + row_idx + " = 0; " + row_idx + " < " + valid_row + "; ++" + row_idx + ") {");
  codegen.Emit("  for (int " + col_idx + " = 0; " + col_idx + " < " + valid_col + "; ++" + col_idx + ") {");
  codegen.Emit("    auto __debug_src_offset = pto::GetTileOffset<decltype(" + src + ")>(" + row_idx + " + (" +
               row_off + "), " + col_idx + " + (" + col_off + "));");
  codegen.Emit("    auto " + debug_val + " = " + src + ".data()[__debug_src_offset];");
  codegen.Emit("    pto::PrintValue(" + debug_val + ", " + col_idx + ");");
  codegen.Emit("  }");
  codegen.Emit("  cce::printf(\"\\n\");");
  codegen.Emit("}");
  return "";
}

// Helper function for binary elementwise operations
static std::string MakeBinaryElementwiseCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Binary elementwise op requires 2 arguments";
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ");");
  return "";
}

// Helper function for binary scalar operations
static std::string MakeBinaryScalarCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Binary scalar op requires 2 arguments";
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ");");
  return "";
}

// Helper function for unary operations
static std::string MakeUnaryCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Unary op requires 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(cce_op_name + "(" + dst + ", " + src + ");");
  return "";
}

// Helper for block.cast - extract target_dtype from kwargs and use TCVT
static std::string MakeBlockCastCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.cast requires 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  int mode = op->GetKwarg<int>("mode");
  // TCVT signature: TCVT(dst, src, rmode)
  // Using default rounding mode (0 for round-to-nearest-even)
  codegen.Emit("TCVT(" + dst + ", " + src + ", " + codegen.GetTypeConverter().ConvertCastRoundMode(mode) +
               ");");
  return "";
}

// Helper for block.cmp/cmps - extract cmp_type from kwargs and use TCMP
static std::string GetCmpModeEnumCCE(int cmp_type) {
  switch (cmp_type) {
    case 0:
      return "CmpMode::EQ";
    case 1:
      return "CmpMode::NE";
    case 2:
      return "CmpMode::LT";
    case 3:
      return "CmpMode::LE";
    case 4:
      return "CmpMode::GT";
    case 5:
      return "CmpMode::GE";
    default:
      CHECK(false) << "Unsupported cmp_type for CCE codegen: " << cmp_type;
      return "CmpMode::EQ";
  }
}

static std::string MakeBlockCmpCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                          codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "block.cmp requires 2 arguments";
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  int cmp_type = op->GetKwarg<int>("cmp_type");
  // signature: TCMP/TCMPS(dst, src0, src1, cmpMode)
  // cmpMode: EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ", " + GetCmpModeEnumCCE(cmp_type) + ");");
  return "";
}

// Helper for block.expands/col_expand - expand scalar/col tile to tile
static std::string MakeBlockExpandsCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "block.expands/col_expand requires 2 arguments";

  std::string src1 = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  // FIX: this instruction is inplaced, dst and target addr should be same
  codegen.Emit(cce_op_name + "(" + dst + ", " + src1 + ");");
  return "";
}

// block.load: emit TASSIGN + TLOAD (same format as original IR layer codegen)
// IR signature: (tensor, offsets_tuple, shapes_tuple) = 3 args
static std::string MakeBlockLoadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "block.load requires 4 arguments: tensor, offsets, shapes, validshape";

  auto src_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
  CHECK(src_tensor_var_ptr != nullptr) << "block.load source tensor must be a Var";

  // Extract offsets tuple
  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "block.load second argument must be a tuple (offsets)";
  CHECK(!offsets_tuple->elements_.empty()) << "block.load offsets tuple must have at least 1 element";

  // Extract shapes tuple
  auto shapes_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple != nullptr) << "block.load third argument must be a tuple (shapes)";

  std::string src_tensor_var = codegen.GetVarName(src_tensor_var_ptr);

  auto src_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(src_tensor_var_ptr->GetType());
  CHECK(src_tensor_type != nullptr) << "block.load source must be TensorType";

  // compute stride-based offset
  std::string offset = ComputeStrideBasedOffset(codegen, src_tensor_var, offsets_tuple, src_tensor_type);

  // Get buffer address from Tensor struct
  std::string src_ptr = codegen.GetPointer(src_tensor_var);
  std::string var_name = codegen.GetCurrentResultTarget();

  codegen.Emit("TASSIGN(" + src_tensor_var + ", " + src_ptr + " + " + offset + ");");
  codegen.Emit("TLOAD(" + var_name + ", " + src_tensor_var + ");");
  return "";
}

// block.store: emit TASSIGN + TSTORE + RegisterOutputPointer (same format as original IR layer codegen)
// IR signature: (tile, offsets_tuple, shapes_tuple, output_tensor) = 4 args
static std::string MakeBlockStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "block.store requires 4 arguments: tile, offsets, shapes, output_tensor";

  std::string src_tile = codegen.GetExprAsCode(op->args_[0]);

  // Extract offsets tuple
  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "block.store second argument must be a tuple (offsets)";
  CHECK(!offsets_tuple->elements_.empty()) << "block.store offsets tuple must have at least 1 element";

  // Extract shapes tuple
  auto shapes_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple != nullptr) << "block.store third argument must be a tuple (shapes)";

  auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[3]);
  CHECK(dst_tensor_var_ptr != nullptr) << "block.store destination tensor must be a Var";

  std::string dst_tensor_var = codegen.GetVarName(dst_tensor_var_ptr);

  auto dst_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(dst_tensor_var_ptr->GetType());
  CHECK(dst_tensor_type != nullptr) << "block.store destination must be TensorType";
  ValidateCCEStoreNZPreconditions("block.store", op->args_[0], offsets_tuple, dst_tensor_type);

  // compute stride-based offset
  std::string offset = ComputeStrideBasedOffset(codegen, dst_tensor_var, offsets_tuple, dst_tensor_type);

  // Get buffer address from Tensor struct
  std::string dst_ptr = codegen.GetPointer(dst_tensor_var);
  std::string var_name = codegen.GetCurrentResultTarget();

  codegen.Emit("TASSIGN(" + dst_tensor_var + ", " + dst_ptr + " + " + offset + ");");
  codegen.Emit("TSTORE(" + dst_tensor_var + ", " + src_tile + ");");
  if (!var_name.empty()) {
    codegen.RegisterOutputPointer(var_name, dst_tensor_var);
    codegen.RegisterOutputTensorStruct(var_name, dst_tensor_var);
    codegen.Emit("auto " + var_name + " = " + dst_tensor_var + ";");
  }
  return "";
}

// Helper function for block.move
static std::string MakeBlockMoveCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.move requires 1 argument: src";

  // Validate memory locations: can't Vec→Vec copies
  auto src_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK(src_type != nullptr) << "Internal error: block.move source must be TileType";
  INTERNAL_CHECK(src_type->memref_.has_value())
      << "Internal error: block.move source TileType must have MemRef (InitMemRef pass should have run)";

  ir::MemorySpace target_memory = op->GetKwarg<ir::MemorySpace>("target_memory");
  ir::MemorySpace src_mem =
      (*src_type->memref_)->memory_space_;  // NOLINT(bugprone-unchecked-optional-access)
  CHECK(!(src_mem == ir::MemorySpace::Vec && target_memory == ir::MemorySpace::Vec))
      << "block.move: Vec to Vec move should use block.vec_move";

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();

  codegen.Emit("TMOV(" + dst + ", " + src + ");");

  return "";
}

// Helper function for block.vec_move (Vec to Vec copy only)
static std::string MakeBlockUbCopyCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.vec_move requires 1 argument: src";

  // Validate memory locations: ONLY support Vec→Vec copies
  auto src_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK(src_type != nullptr) << "Internal error: block.vec_move source must be TileType";
  INTERNAL_CHECK(src_type->memref_.has_value())
      << "Internal error: block.vec_move source TileType must have MemRef (InitMemRef pass should have run)";

  // Verify source is on Vec
  ir::MemorySpace src_mem =
      (*src_type->memref_)->memory_space_;  // NOLINT(bugprone-unchecked-optional-access)
  CHECK(src_mem == ir::MemorySpace::Vec)
      << "block.vec_move: source must be on Vec memory, got " << ir::MemorySpaceToString(src_mem);

  // Get source and destination expressions
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();

  // Emit TMOV instruction for Vec→Vec copy
  codegen.Emit("TMOV(" + dst + ", " + src + ");");

  return "";
}

// Helper function for block.alloc (no-op: allocation handled elsewhere)
static std::string MakeBlockAllocCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No C++ emission - MemRef/Tile setup handled in prologue
}

// Helper function for block.get_block_idx (returns value expression)
static std::string MakeBlockGetBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  CHECK(op->args_.size() == 0) << "block.get_block_idx requires no arguments";
  return "get_block_idx()";
}

// Helper function for block.make_tile (no-op: allocation handled elsewhere)
static std::string MakeBlockCreateTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No C++ emission - Tile declaration handled in prologue
}

// Helper function for block.full
static std::string MakeBlockFullCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string scalar = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TEXPANDS(" + dst + ", " + scalar + ");");
  return "";
}

// ============================================================================
// Matmul Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.matmul")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) -> std::string {
      CHECK(op->args_.size() == 2) << "block.matmul requires 2 arguments: lhs, rhs";

      std::string lhs = codegen.GetExprAsCode(op->args_[0]);
      std::string rhs = codegen.GetExprAsCode(op->args_[1]);
      std::string dst = codegen.GetCurrentResultTarget();

      codegen.Emit("TMATMUL(" + dst + ", " + lhs + ", " + rhs + ");");

      return "";  // Statement-emitting mode
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.matmul_acc")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) -> std::string {
      CHECK(op->args_.size() == 3) << "block.matmul_acc requires 3 arguments: acc, lhs, rhs";

      [[maybe_unused]] std::string acc = codegen.GetExprAsCode(op->args_[0]);
      std::string lhs = codegen.GetExprAsCode(op->args_[1]);
      std::string rhs = codegen.GetExprAsCode(op->args_[2]);
      std::string dst = codegen.GetCurrentResultTarget();

      // TMATMUL_ACC accumulates into dst, which should be initialized from acc
      // In CCE ISA, this is typically: TMATMUL_ACC(dst, acc, lhs, rhs)
      codegen.Emit("TMATMUL_ACC(" + dst + ", " + acc + ", " + lhs + ", " + rhs + ");");

      return "";  // Statement-emitting mode
    });

// ============================================================================
// Elementwise Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.maximum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TMAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.minimum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TMIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TMULS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TADDS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.divs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TDIVS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.subs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TSUBS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.cmp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCmpCodegenCCE("TCMP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.cmps")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCmpCodegenCCE("TCMPS", op, codegen);
    });

// ============================================================================
// Unary Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TEXP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TNEG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.recip")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRECIP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.rsqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TLOG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TABS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRELU", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCastCodegenCCE(op, codegen);
    });

// ============================================================================
// Memory Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.alloc")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockAllocCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.make_tile")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCreateTileCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.load")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockLoadCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.store")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockStoreCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.move")
    .set_pipe(ir::PipeType::MTE1)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockMoveCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.vec_move")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockUbCopyCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.get_block_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockGetBlockIdxCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockFullCodegenCCE(op, codegen);
    });

// ============================================================================
// Reduction Operations
// ============================================================================

static std::string MakeBlockRowReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "TROW" << op_prefix << " requires 2 arguments";
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string tmp_tile = codegen.GetExprAsCode(op->args_[1]);
  std::string result = codegen.GetCurrentResultTarget();

  codegen.Emit("TROW" + op_prefix + "(" + result + ", " + tile + ", " + tmp_tile + ");");
  return "";
}

static std::string MakeBlockColReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "TCOL" << op_prefix << " requires 1 argument";
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string result = codegen.GetCurrentResultTarget();

  codegen.Emit("TCOL" + op_prefix + "(" + result + ", " + tile + ");");
  return "";
}

// Helper function for reduction operations (sum, max)
static std::string MakeBlockReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  int axis = op->GetKwarg<int>("axis");
  if (axis == 0) {
    return MakeBlockColReductionCodegenCCE(op_prefix, op, codegen_base);
  } else {
    return MakeBlockRowReductionCodegenCCE(op_prefix, op, codegen_base);
  }
}

REGISTER_BACKEND_OP(Backend910B_CCE, "block.sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockReductionCodegenCCE("SUM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockReductionCodegenCCE("MAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockRowReductionCodegenCCE("SUM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockRowReductionCodegenCCE("MAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockReductionCodegenCCE("MIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockRowReductionCodegenCCE("MIN", op, codegen);
    });

// ============================================================================
// Broadcast Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.fillpad")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TFILLPAD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockExpandsCodegenCCE("TCOLEXPAND", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.expands")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockExpandsCodegenCCE("TEXPANDS", op, codegen);
    });

// ============================================================================
// Transform Operations (view/reshape/transpose: same buffer, reinterpret)
// ============================================================================

static std::string MakeBlockTransformCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() >= 1) << "block view/reshape/transpose require at least 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit("TMOV(" + dst + ", " + src + ");");
  return "";
}

static std::string MakeTileReshapeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string target_var = codegen.GetCurrentResultTarget();
  std::string input_var = codegen.GetExprAsCode(op->args_[0]);

  codegen.Emit("TRESHAPE(" + target_var + ", " + input_var + ");");
  return "";
}

static std::string MakeTileTransposeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string target_var = codegen.GetCurrentResultTarget();
  std::string input_var = codegen.GetExprAsCode(op->args_[0]);
  auto axis1 = codegen.GetConstIntValue(op->args_[1]);
  auto axis2 = codegen.GetConstIntValue(op->args_[2]);
  size_t ndim = ir::As<ir::TileType>(op->args_[0]->GetType())->shape_.size();

  INTERNAL_CHECK(ndim == 2) << "Codegen only supports 2D tiles, but got " << ndim << "D tile";
  INTERNAL_CHECK(axis1 != axis2) << "tile.transpose: axis1 and axis2 must be different, but got axis1=axis2="
                                 << axis1;
  INTERNAL_CHECK(axis1 >= 0 && axis1 < ndim && axis2 >= 0 && axis2 < ndim)
      << "tile.transpose: axis1 and axis2 must be in range [0, " << ndim << "), but got axis1=" << axis1
      << ", axis2=" << axis2;

  codegen.Emit("TTRANS(" + target_var + ", " + input_var + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "block.reshape")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileReshapeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.transpose")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileReshapeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.view")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockTransformCodegenCCE(op, codegen);
    });

// ============================================================================
// Sync / Barrier Operations (inserted by insert_sync_pass)
// ============================================================================

static std::string PipeTypeToCCEString(ir::PipeType pipe) {
  switch (pipe) {
    case ir::PipeType::MTE1:
      return "PIPE_MTE1";
    case ir::PipeType::MTE2:
      return "PIPE_MTE2";
    case ir::PipeType::MTE3:
      return "PIPE_MTE3";
    case ir::PipeType::M:
      return "PIPE_M";
    case ir::PipeType::V:
      return "PIPE_V";
    case ir::PipeType::S:
      return "PIPE_S";
    case ir::PipeType::FIX:
      return "PIPE_FIX";
    case ir::PipeType::ALL:
      return "PIPE_ALL";
    default:
      return "PIPE_V";
  }
}

static std::string MakeSyncCodegenCCE(const std::string& isa_name, const ir::CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
  auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
  int event_id = op->GetKwarg<int>("event_id");
  std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
  std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
  std::string event_id_str = "EVENT_ID" + std::to_string(event_id);
  codegen.Emit(isa_name + "(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id_str + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_src")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncCodegenCCE("set_flag", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_dst")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncCodegenCCE("wait_flag", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_v")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
      if (codegen.GetArch() == "a3") {
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_V);");
      }
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_m")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_M);");
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_ALL);");
      return "";
    });

// ============================================================================
// Cross-core Sync Operations
// ============================================================================

// Cross-core SET: ffts_cross_core_sync(PIPE_xxx, getFFTSMsg(FFTS_MODE_VAL, event_id))
// Cross-core WAIT: wait_flag_dev(event_id)
// SET signals completion from a pipe; WAIT blocks until the other core signals.

static std::string MakeCrossCoreSetCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                               bool is_dynamic) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto pipe = op->GetKwarg<int>("pipe");
  std::string pipe_str = PipeTypeToCCEString(static_cast<ir::PipeType>(pipe));
  bool is_a5 = (codegen.GetArch() == "a5");
  if (is_a5) {
    // a5: CUBE side sets for BOTH vector subcores (v0: id, v1: id+16)
    //     VEC side sets for CUBE with a single set
    if (codegen.IsInCubeSection()) {
      // CUBE setting for VEC: auto-expand to two set_intra_block calls
      if (is_dynamic) {
        std::string event_id = codegen.GetExprAsCode(op->args_[0]);
        codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + ");");
        codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + " + 16);");
      } else {
        int event_id = op->GetKwarg<int>("event_id");
        codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
        codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id + 16) + ");");
      }
    } else {
      // VEC setting for CUBE: single set
      if (is_dynamic) {
        std::string event_id = codegen.GetExprAsCode(op->args_[0]);
        codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + ");");
      } else {
        int event_id = op->GetKwarg<int>("event_id");
        codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
      }
    }
  } else {
    if (is_dynamic) {
      std::string event_id = codegen.GetExprAsCode(op->args_[0]);
      codegen.Emit("ffts_cross_core_sync(" + pipe_str + ", getFFTSMsg(FFTS_MODE_VAL, " + event_id + "));");
    } else {
      int event_id = op->GetKwarg<int>("event_id");
      codegen.Emit("ffts_cross_core_sync(" + pipe_str + ", getFFTSMsg(FFTS_MODE_VAL, " +
                    std::to_string(event_id) + "));");
    }
  }
  return "";
}

static std::string MakeCrossCoreWaitCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                                bool is_dynamic) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto pipe = op->GetKwarg<int>("pipe");
  std::string pipe_str = PipeTypeToCCEString(static_cast<ir::PipeType>(pipe));
  bool is_a5 = (codegen.GetArch() == "a5");
  if (is_a5) {
    // a5: CUBE side waits for BOTH vector subcores (v0: id, v1: id+16)
    //     VEC side waits for CUBE with a single wait
    if (codegen.IsInCubeSection()) {
      // CUBE waiting for VEC: auto-expand to two wait_intra_block calls
      if (is_dynamic) {
        std::string event_id = codegen.GetExprAsCode(op->args_[0]);
        codegen.Emit("wait_intra_block(" + pipe_str + ", " + event_id + ");");
        codegen.Emit("wait_intra_block(" + pipe_str + ", " + event_id + " + 16);");
      } else {
        int event_id = op->GetKwarg<int>("event_id");
        codegen.Emit("wait_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
        codegen.Emit("wait_intra_block(" + pipe_str + ", " + std::to_string(event_id + 16) + ");");
      }
    } else {
      // VEC waiting for CUBE: single wait
      if (is_dynamic) {
        std::string event_id = codegen.GetExprAsCode(op->args_[0]);
        codegen.Emit("wait_intra_block(" + pipe_str + ", " + event_id + ");");
      } else {
        int event_id = op->GetKwarg<int>("event_id");
        codegen.Emit("wait_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
      }
    }
  } else {
    if (is_dynamic) {
      std::string event_id = codegen.GetExprAsCode(op->args_[0]);
      codegen.Emit("wait_flag_dev(" + event_id + ");");
    } else {
      int event_id = op->GetKwarg<int>("event_id");
      codegen.Emit("wait_flag_dev(" + std::to_string(event_id) + ");");
    }
  }
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.set_cross_core")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCrossCoreSetCodegenCCE(op, codegen, false);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.wait_cross_core")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCrossCoreWaitCodegenCCE(op, codegen, false);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.set_cross_core_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCrossCoreSetCodegenCCE(op, codegen, true);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.wait_cross_core_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCrossCoreWaitCodegenCCE(op, codegen, true);
    });

// ============================================================================
// Dynamic event_id sync operations
// ============================================================================

static std::string MakeSyncSrcDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
  auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
  std::string event_id = codegen.GetExprAsCode(op->args_[0]);
  std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
  std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
  // EventId array access (contains '[') already returns event_t �?no cast needed
  if (event_id.find('[') != std::string::npos) {
    codegen.Emit("set_flag(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id + ");");
  } else {
    codegen.Emit("set_flag(" + set_pipe_str + ", " + wait_pipe_str + ", (event_t)" + event_id + ");");
  }
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_src_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncSrcDynCodegenCCE(op, codegen);
    });

static std::string MakeSyncDstDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
  auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
  std::string event_id = codegen.GetExprAsCode(op->args_[0]);
  std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
  std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
  // EventId array access (contains '[') already returns event_t �?no cast needed
  if (event_id.find('[') != std::string::npos) {
    codegen.Emit("wait_flag(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id + ");");
  } else {
    codegen.Emit("wait_flag(" + set_pipe_str + ", " + wait_pipe_str + ", (event_t)" + event_id + ");");
  }
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_dst_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncDstDynCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "debug.dump_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeDebugDumpTensorCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "debug.dump_tile")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeDebugDumpTileCodegenCCE(op, codegen);
    });

// ============================================================================
// Missing block operations: get_block_num, get_subblock_idx, index_cast
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.get_block_num")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      CHECK(op->args_.size() == 0) << "block.get_block_num requires no arguments";
      return std::string("get_block_num()");
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.get_subblock_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      CHECK(op->args_.size() == 0) << "block.get_subblock_idx requires no arguments";
      return std::string("get_subblockid()");
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.index_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
      CHECK(op->args_.size() == 1) << "block.index_cast requires 1 argument";
      std::string value = codegen.GetExprAsCode(op->args_[0]);
      return "(int32_t)(" + value + ")";
    });

static std::string MakeTensorDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string target_var = codegen.GetCurrentResultTarget();
  int64_t axis = codegen.GetConstIntValue(op->args_[1]);

  auto input_tensor = ir::As<ir::TensorType>(op->args_[0]->GetType());
  CHECK(input_tensor) << "tensor.dim need TensorType for first arg, but got "
                      << op->args_[0]->GetType()->TypeName();
  auto ndims = static_cast<int64_t>(input_tensor->shape_.size());
  int64_t pad_dims = 5 - ndims;  // pto-isa pad shape to 5 dims

  // get axis in GlobalTensor 5 dims
  if (axis < 0) {
    axis += ndims;
  }
  int64_t gt_dim = pad_dims + axis;

  // get GlobalTensor of input_tensor
  auto input_tensor_var = ir::As<ir::Var>(op->args_[0]);
  CHECK(input_tensor_var) << "tensor.dim need var with TensorType for first arg";
  std::string input_tensor_var_name = codegen.GetVarName(input_tensor_var);

  codegen.Emit("int " + target_var + " = " + input_tensor_var_name + ".GetShape(GlobalTensorDim::DIM_" +
               std::to_string(gt_dim) + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "tensor.dim")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorDimCodegenCCE(op, codegen);
    });

// ============================================================================
// Block GetVal/SetVal Operations
// ============================================================================

static std::string MakeBlockGetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "block.getval requires 2 arguments (tile, index), but got "
                                << op->args_.size();

  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string index = codegen.GetExprAsCode(op->args_[1]);

  // Return the expression, framework will handle assignment
  return tile + ".GetValue(" + index + ")";
}

static std::string MakeBlockSetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "block.setval requires 3 arguments (tile, index, value), but got "
                                << op->args_.size();

  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string index = codegen.GetExprAsCode(op->args_[1]);
  std::string value = codegen.GetExprAsCode(op->args_[2]);

  codegen.Emit(tile + ".SetValue(" + index + ", " + value + ");");

  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "block.getval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockGetValCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.setval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockSetValCodegenCCE(op, codegen);
    });

// ============================================================================
// Tensor GetVal/SetVal Operations
// ============================================================================

static std::string MakeTensorGetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tensor.getval requires 2 arguments (tensor, offset), but got "
                                << op->args_.size();

  auto tensor_var = ir::As<ir::Var>(op->args_[0]);
  INTERNAL_CHECK(tensor_var) << "tensor.getval requires tensor to be a Var";
  auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
  INTERNAL_CHECK(tensor_type) << "tensor.getval requires TensorType";
  std::string tensor_name = codegen.GetVarName(tensor_var);
  std::string offset = codegen.GetExprAsCode(op->args_[1]);
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);

  std::string tensor_ptr = codegen.GetPointer(tensor_name);
  if (tensor_ptr.empty()) {
    tensor_ptr = tensor_name + ".data()";
  }

  // Return the expression, framework will handle assignment
  return "*((__gm__ " + dtype_str + "*)" + tensor_ptr + " + " + offset + ")";
}

static std::string MakeTensorSetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "tensor.setval requires 3 arguments (tensor, offset, value), but got "
                                << op->args_.size();

  auto tensor_var = ir::As<ir::Var>(op->args_[0]);
  INTERNAL_CHECK(tensor_var) << "tensor.setval requires tensor to be a Var";
  std::string tensor_name = codegen.GetVarName(tensor_var);
  std::string offset = codegen.GetExprAsCode(op->args_[1]);
  std::string value = codegen.GetExprAsCode(op->args_[2]);

  auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
  INTERNAL_CHECK(tensor_type) << "tensor.setval requires TensorType";
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);

  std::string tensor_ptr = codegen.GetPointer(tensor_name);
  if (tensor_ptr.empty()) {
    tensor_ptr = tensor_name + ".data()";
  }

  codegen.Emit("*((__gm__ " + dtype_str + "*)" + tensor_ptr + " + " + offset + ") = " + value + ";");

  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "tensor.getval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorGetValCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tensor.setval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorSetValCodegenCCE(op, codegen);
    });

static std::string MakeTensorReadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tensor.read requires 2 arguments (tensor, indices)";

  auto tensor_var = ir::As<ir::Var>(op->args_[0]);
  CHECK(tensor_var) << "tensor.read first argument must be a Var";
  auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
  CHECK(tensor_type) << "tensor.read first argument must be TensorType";

  auto indices_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
  CHECK(indices_tuple) << "tensor.read second argument must be a MakeTuple of indices";

  std::string tensor_name = codegen.GetVarName(tensor_var);
  std::string base_ptr = codegen.GetPointer(tensor_name);
  if (base_ptr.empty()) {
    base_ptr = tensor_name + ".data()";
  }

  const auto& indices = indices_tuple->elements_;
  const auto& shape = tensor_type->shape_;

  std::ostringstream idx_oss;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i > 0) idx_oss << " + ";
    idx_oss << codegen.GetExprAsCode(indices[i]);
    for (size_t j = i + 1; j < shape.size(); ++j) {
      idx_oss << " * " << codegen.GetExprAsCode(shape[j]);
    }
  }
  std::string idx_expr = idx_oss.str();

  // Return expression value (let AssignStmt handle the declaration)
  // In single-file mode, base_ptr is already a __gm__ pointer parameter.
  // Directly index into it without casting (casting from __gm__ to local is not allowed).
  return base_ptr + "[" + idx_expr + "]";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "tensor.read")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorReadCodegenCCE(op, codegen);
    });

// ============================================================================
// Mutex (Buffer-ID Token) — A5 CCE Codegen
// ----------------------------------------------------------------------------
// Lowers system.mutex_lock/unlock to CCE intrinsics get_buf/rls_buf.
// API: get_buf(PIPE_MTE2, mutexId, 0);  rls_buf(PIPE_MTE2, mutexId, 0);
// ============================================================================

static std::string MakeMutexLockCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("pipe"));
  auto mutex_id = op->GetKwarg<int>("mutex_id");
  if (codegen.ShouldSkipVPipeMutex(pipe, {mutex_id})) return "";
  int mode = 0;
  for (const auto& [key, value] : op->kwargs_) {
    if (key == "mode") mode = std::any_cast<int>(value);
  }
  std::string pipe_str = PipeTypeToCCEString(pipe);
  codegen.Emit("get_buf(" + pipe_str + ", " + std::to_string(mutex_id) + ", " + std::to_string(mode) + ");");
  return "";
}

static std::string MakeMutexUnlockCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("pipe"));
  auto mutex_id = op->GetKwarg<int>("mutex_id");
  if (codegen.ShouldSkipVPipeMutex(pipe, {mutex_id})) return "";
  int mode = 0;
  for (const auto& [key, value] : op->kwargs_) {
    if (key == "mode") mode = std::any_cast<int>(value);
  }
  std::string pipe_str = PipeTypeToCCEString(pipe);
  codegen.Emit("rls_buf(" + pipe_str + ", " + std::to_string(mutex_id) + ", " + std::to_string(mode) + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.mutex_lock")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeMutexLockCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.mutex_unlock")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeMutexUnlockCodegenCCE(op, codegen);
    });

// Dynamic mutex_id: the mutex_id expression is already a runtime value
// (e.g. from tuple-index if-else chain). get_buf/rls_buf take runtime args.
static std::vector<int> GetBufIdValuesFromKwargs(const ir::CallPtr& op) {
  std::vector<int> values;
  for (const auto& [key, value] : op->kwargs_) {
    if (key == "buf_id_values") {
      values = std::any_cast<std::vector<int>>(value);
      return values;
    }
  }
  int max_id = 2;
  for (const auto& [key, value] : op->kwargs_) {
    if (key == "max_mutex_id") max_id = std::any_cast<int>(value);
  }
  for (int i = 0; i < max_id; ++i) values.push_back(i);
  return values;
}

static std::string MakeMutexLockDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("pipe"));
  if (codegen.ShouldSkipVPipeMutex(pipe, GetBufIdValuesFromKwargs(op))) return "";
  int mode = 0;
  for (const auto& [key, value] : op->kwargs_) {
    if (key == "mode") mode = std::any_cast<int>(value);
  }
  std::string pipe_str = PipeTypeToCCEString(pipe);
  std::string mutex_id_expr = codegen.GetExprAsCode(op->args_[0]);
  codegen.Emit("get_buf(" + pipe_str + ", " + mutex_id_expr + ", " + std::to_string(mode) + ");");
  return "";
}

static std::string MakeMutexUnlockDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("pipe"));
  if (codegen.ShouldSkipVPipeMutex(pipe, GetBufIdValuesFromKwargs(op))) return "";
  int mode = 0;
  for (const auto& [key, value] : op->kwargs_) {
    if (key == "mode") mode = std::any_cast<int>(value);
  }
  std::string pipe_str = PipeTypeToCCEString(pipe);
  std::string mutex_id_expr = codegen.GetExprAsCode(op->args_[0]);
  codegen.Emit("rls_buf(" + pipe_str + ", " + mutex_id_expr + ", " + std::to_string(mode) + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.mutex_lock_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeMutexLockDynCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.mutex_unlock_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeMutexUnlockDynCodegenCCE(op, codegen);
    });

}  // namespace backend
}  // namespace pypto

