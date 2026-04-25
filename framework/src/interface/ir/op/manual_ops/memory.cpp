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
 * @file manual_ops/memory.cpp
 * @brief Manual (non-SSA) memory operations: load, store, store_fp, move, ub_copy, full, fillpad,
 * fillpad_inplace, fillpad_expand.
 *
 * Each "manual" op receives the pre-allocated output tile as its last argument
 * and returns that tile's type rather than creating a fresh SSA result type.
 * This mirrors the hardware semantics where the programmer explicitly manages
 * tile buffers.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

/// Return the TileType of the last argument (the pre-allocated output tile).
static TypePtr DeduceManualOutTileType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& op_name, size_t expected_args) {
  CHECK(args.size() == expected_args)
      << "The operator " << op_name << " requires exactly " << expected_args << " arguments, but got "
      << args.size();
  auto out_type = As<TileType>(args.back()->GetType());
  CHECK(out_type) << "The operator " << op_name
                  << " requires last argument (out) to be TileType, but got "
                  << args.back()->GetType()->TypeName();
  return out_type;
}

static TypePtr DeduceManualFillPadType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& op_name, bool allow_expand,
                                       bool require_shared_backing_storage = false) {
  auto out_type = As<TileType>(DeduceManualOutTileType(args, kwargs, op_name, 2));
  CHECK(out_type) << op_name << ": out must be TileType";
  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << op_name << ": src must be TileType";
  CHECK(out_type->tile_view_.has_value())
      << op_name << ": out tile must carry tile_view metadata";

  TileView view = out_type->tile_view_.value();
  int pad_value = static_cast<int>(view.pad);
  CHECK(pad_value >= static_cast<int>(TilePad::null) && pad_value <= static_cast<int>(TilePad::min))
      << op_name << ": out.tile_view.pad must be one of TilePad.null/zero/max/min";
  CHECK(pad_value != static_cast<int>(TilePad::null))
      << op_name << ": out.tile_view.pad must not be TilePad.null";

  if (require_shared_backing_storage) {
    CHECK(src_type->memref_.has_value() && out_type->memref_.has_value())
        << op_name << ": src and out must share backing storage";
    auto src_memref = src_type->memref_.value();
    auto out_memref = out_type->memref_.value();
    CHECK(src_memref->memory_space_ == out_memref->memory_space_ &&
          structural_equal(src_memref->addr_, out_memref->addr_))
        << op_name << ": src and out must share backing storage";

    CHECK(src_type->shape_.size() == 2 && out_type->shape_.size() == 2)
        << op_name << ": src/out tile shapes must be rank-2";
    auto src_rows = As<ConstInt>(src_type->shape_[0]);
    auto src_cols = As<ConstInt>(src_type->shape_[1]);
    auto out_rows = As<ConstInt>(out_type->shape_[0]);
    auto out_cols = As<ConstInt>(out_type->shape_[1]);
    CHECK(src_rows && src_cols && out_rows && out_cols)
        << op_name << ": src/out tile shapes must be static";
    CHECK(out_rows->value_ == src_rows->value_ && out_cols->value_ == src_cols->value_)
        << op_name << ": src and out tile rows/cols must match";
  }

  if (allow_expand) {
    CHECK(src_type->shape_.size() == 2 && out_type->shape_.size() == 2)
        << op_name << ": src/out tile shapes must be rank-2";

    auto src_rows = As<ConstInt>(src_type->shape_[0]);
    auto src_cols = As<ConstInt>(src_type->shape_[1]);
    auto out_rows = As<ConstInt>(out_type->shape_[0]);
    auto out_cols = As<ConstInt>(out_type->shape_[1]);
    CHECK(src_rows && src_cols && out_rows && out_cols)
        << op_name << ": src/out tile shapes must be static";
    CHECK(out_rows->value_ >= src_rows->value_ && out_cols->value_ >= src_cols->value_)
        << op_name << ": out tile rows/cols must be >= src tile rows/cols";
  }
  return out_type;
}

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

// manual.load: (tensor, offsets, out) -> TileType (out's type)
REGISTER_OP("manual.load")
    .set_op_category("ManualOp")
    .set_description(
        "Manual load: copy data from a global tensor into a pre-allocated tile. "
        "The output tile (last arg) defines the destination buffer; its type is returned. "
        "Partition view sizes are derived from the tile type's shape/valid_shape.")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("offsets", "Offset tuple per dimension (MakeTuple)")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .set_attr<std::string>("layout")
    .set_attr<std::vector<int>>("tile_dims")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 3) << "manual.load requires 3 arguments, got " << args.size();
      CHECK(As<TensorType>(args[0]->GetType()))
          << "manual.load: arg 0 must be TensorType";
      auto offsets = As<MakeTuple>(args[1]);
      CHECK(offsets) << "manual.load: arg 1 must be MakeTuple (offsets)";
      CHECK(As<TileType>(args[2]->GetType()))
          << "manual.load: arg 2 must be TileType";
      return args[2]->GetType();
    });

// manual.store: (tile, offsets, output_tensor) -> TensorType
REGISTER_OP("manual.store")
    .set_op_category("ManualOp")
    .set_description(
        "Manual store: copy data from a pre-allocated tile to a global tensor.")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("offsets", "Offset tuple per dimension (MakeTuple)")
    .add_argument("output_tensor", "Destination tensor (TensorType)")
    .set_attr<std::vector<int>>("tile_dims")
    .set_attr<std::string>("relu_pre_mode")
    .set_attr<int>("pre_quant_scalar")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 3) << "manual.store requires 3 arguments, got " << args.size();
      CHECK(As<TileType>(args[0]->GetType()))
          << "manual.store: arg 0 must be TileType";
      auto offsets = As<MakeTuple>(args[1]);
      CHECK(offsets) << "manual.store: arg 1 must be MakeTuple (offsets)";
      auto out_type = As<TensorType>(args[2]->GetType());
      CHECK(out_type) << "manual.store: arg 2 must be TensorType";
      return out_type;
    });

// manual.store_fp: (tile, fp_tile, offsets, output_tensor) -> TensorType
REGISTER_OP("manual.store_fp")
    .set_op_category("ManualOp")
    .set_description(
        "Manual floating-point store: copy data from a pre-allocated Acc tile to a global tensor "
        "using an auxiliary fp tile.")
    .add_argument("tile", "Source tile (TileType, Acc memory)")
    .add_argument("fp_tile", "Floating-point parameter tile (TileType)")
    .add_argument("offsets", "Offset tuple per dimension (MakeTuple)")
    .add_argument("output_tensor", "Destination tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 4) << "manual.store_fp requires 4 arguments, got " << args.size();
      CHECK(As<TileType>(args[0]->GetType()))
          << "manual.store_fp: arg 0 must be TileType";
      CHECK(As<TileType>(args[1]->GetType()))
          << "manual.store_fp: arg 1 must be TileType";
      auto offsets = As<MakeTuple>(args[2]);
      CHECK(offsets) << "manual.store_fp: arg 2 must be MakeTuple (offsets)";
      auto out_type = As<TensorType>(args[3]->GetType());
      CHECK(out_type) << "manual.store_fp: arg 3 must be TensorType";
      return out_type;
    });

// manual.move: (src_tile, out) -> TileType (out's type)
REGISTER_OP("manual.move")
    .set_op_category("ManualOp")
    .set_description(
        "Manual move: transfer a tile between memory levels into a pre-allocated buffer. "
        "The TMOV variant is determined by the output tile's memory space.")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .set_attr<std::string>("acc_to_vec_mode")
    .set_attr<std::string>("relu_pre_mode")
    .set_attr<int>("pre_quant_scalar")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2)
          << "The operator manual.move requires 2 arguments, but got " << args.size();
      auto out_type = As<TileType>(args.back()->GetType());
      CHECK(out_type) << "manual.move: last argument (out) must be TileType";
      return out_type;
    });

// manual.move_fp: (src_tile, fp_tile, out) -> TileType (out's type)
REGISTER_OP("manual.move_fp")
    .set_op_category("ManualOp")
    .set_description(
        "Manual move with scaling tile: convert an Acc tile into a pre-allocated Vec tile.")
    .add_argument("src", "Source tile (TileType, Acc memory)")
    .add_argument("fp_tile", "Floating-point parameter tile (TileType, Scaling memory)")
    .add_argument("out", "Pre-allocated destination tile (TileType, Vec memory)")
    .set_attr<std::string>("acc_to_vec_mode")
    .set_attr<std::string>("relu_pre_mode")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 3)
          << "The operator manual.move_fp requires 3 arguments, but got " << args.size();
      CHECK(As<TileType>(args[0]->GetType()))
          << "manual.move_fp: arg 0 must be TileType";
      CHECK(As<TileType>(args[1]->GetType()))
          << "manual.move_fp: arg 1 must be TileType";
      auto out_type = As<TileType>(args.back()->GetType());
      CHECK(out_type) << "manual.move_fp: last argument (out) must be TileType";
      return out_type;
    });

// manual.insert: (src, index_row, index_col, out) or (src, index_row, index_col, offset, out) -> TileType
REGISTER_OP("manual.insert")
    .set_op_category("ManualOp")
    .set_description(
        "Manual insert: insert source sub-tile into destination tile at (indexRow, indexCol). "
        "Corresponds to pto-isa TINSERT instruction for UB→L1 transfer.")
    .add_argument("src", "Source sub-tile (TileType, Vec memory)")
    .add_argument("index_row", "Row index where insertion begins")
    .add_argument("index_col", "Column index where insertion begins")
    .add_argument("out", "Destination tile (TileType, Mat memory), or offset + out when 5 args")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 4 || args.size() == 5)
          << "The operator manual.insert requires 4 or 5 arguments, but got " << args.size();
      auto out_type = As<TileType>(args.back()->GetType());
      CHECK(out_type) << "manual.insert: last argument (out) must be TileType";
      return out_type;
    });

// manual.ub_copy: (src_tile, out) -> TileType (out's type)
REGISTER_OP("manual.ub_copy")
    .set_op_category("ManualOp")
    .set_description(
        "Manual UB-to-UB copy: copy a tile within unified buffer into a pre-allocated buffer.")
    .add_argument("src", "Source UB tile (TileType)")
    .add_argument("out", "Pre-allocated destination UB tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.ub_copy", 2);
    });

// manual.full: (scalar, out) -> TileType (out's type)
// Fills the pre-allocated tile with a scalar value. Shape comes from out's TileType.
REGISTER_OP("manual.full")
    .set_op_category("ManualOp")
    .set_description(
        "Manual fill: broadcast a scalar value across a pre-allocated tile (out = scalar).")
    .add_argument("scalar", "Fill value (ScalarType or constant)")
    .add_argument("out", "Pre-allocated tile to fill (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.full", 2);
    });

// manual.fillpad: (src_tile, out) -> TileType (out's type)
REGISTER_OP("manual.fillpad")
    .set_op_category("ManualOp")
    .set_description(
        "Manual fill-with-padding: copy src tile into out and pad remaining elements.")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualFillPadType(args, kwargs, "manual.fillpad", false);
    });

// manual.fillpad_inplace: (src_tile, out) -> TileType (out's type)
REGISTER_OP("manual.fillpad_inplace")
    .set_op_category("ManualOp")
    .set_description(
        "Manual inplace fill-with-padding: src and out share backing storage while preserving distinct metadata.")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualFillPadType(args, kwargs, "manual.fillpad_inplace", false, true);
    });

// manual.fillpad_expand: (src_tile, out) -> TileType (out's type)
REGISTER_OP("manual.fillpad_expand")
    .set_op_category("ManualOp")
    .set_description(
        "Manual fill-with-padding: copy src tile into a larger out tile and pad remaining elements.")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualFillPadType(args, kwargs, "manual.fillpad_expand", true);
    });

// manual.set_validshape: (row, col, tile) -> TileType (tile's type)
REGISTER_OP("manual.set_validshape")
    .set_op_category("ManualOp")
    .set_description(
        "Update valid-shape metadata on a dynamic tile in place. "
        "Emits a pto.set_validshape instruction to set the runtime valid row/col.")
    .add_argument("row", "Runtime valid row count (ScalarType or constant)")
    .add_argument("col", "Runtime valid column count (ScalarType or constant)")
    .add_argument("tile", "Dynamic tile buffer to update (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.set_validshape", 3);
    });

}  // namespace ir
}  // namespace pypto
