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
 * @file manual_ops/sort.cpp
 * @brief Manual (non-SSA) sorting operations: sort32, mrgsort (format1), mrgsort2 (format2).
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// ---------------------------------------------------------------------------
// sort32
// ---------------------------------------------------------------------------

REGISTER_OP("manual.sort32")
    .set_op_category("ManualOp")
    .set_description(
        "Manual sort fixed-size 32-element blocks with index mapping. "
        "Sorts each 32-element block and produces sorted values and permutation indices.")
    .add_argument("src", "Input tile (TileType)")
    .add_argument("idx", "Input/output tile for permutation indices (TileType, UINT32)")
    .add_argument("dst", "Output tile for sorted values (TileType)")
    .add_argument("tmp", "Optional scratch tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 3 || args.size() == 4)
          << "manual.sort32 requires 3 or 4 arguments (src, idx, dst[, tmp]), but got "
          << args.size();
      auto dst_type = As<TileType>(args[2]->GetType());
      CHECK(dst_type) << "manual.sort32: dst must be TileType, but got "
                      << args[2]->GetType()->TypeName();
      return dst_type;
    });

// ---------------------------------------------------------------------------
// mrgsort (format1): single tile, block-wise merge
// ---------------------------------------------------------------------------

REGISTER_OP("manual.mrgsort")
    .set_op_category("ManualOp")
    .set_description(
        "Manual merge sort on sorted lists (format1). "
        "Performs merge sort with specified block length.")
    .add_argument("src", "Input tile (TileType)")
    .add_argument("dst", "Output tile (TileType)")
    .set_attr<int>("block_len")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2)
          << "manual.mrgsort requires 2 arguments (src, dst) for format1, but got "
          << args.size();
      auto dst_type = As<TileType>(args[1]->GetType());
      CHECK(dst_type) << "manual.mrgsort: dst must be TileType, but got "
                      << args[1]->GetType()->TypeName();
      return dst_type;
    });

// ---------------------------------------------------------------------------
// mrgsort2 (format2): multi-list top-K merge (2, 3, or 4 sources)
// ---------------------------------------------------------------------------

REGISTER_OP("manual.mrgsort2")
    .set_op_category("ManualOp")
    .set_description(
        "Manual merge sort on multiple sorted lists (format2). "
        "Merges 2, 3, or 4 pre-sorted source tiles into dst using tmp buffer.")
    .add_argument("src0", "First input tile (TileType)")
    .add_argument("dst", "Output tile (TileType)")
    .add_argument("tmp", "Temporary buffer tile (TileType, same shape as dst)")
    .add_argument("src1", "Second input tile (TileType)")
    .set_attr<bool>("exhausted")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() >= 4 && args.size() <= 6)
          << "manual.mrgsort2 requires 4-6 arguments (src0, dst, tmp, src1, [src2, src3]), but got "
          << args.size();
      auto dst_type = As<TileType>(args[1]->GetType());
      CHECK(dst_type) << "manual.mrgsort2: dst must be TileType, but got "
                      << args[1]->GetType()->TypeName();
      return dst_type;
    });

}  // namespace ir
}  // namespace pypto
