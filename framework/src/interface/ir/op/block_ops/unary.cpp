/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "codegen/cce/cce_codegen.h"
#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/pipe.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockUnaryType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  INTERNAL_CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                          << args.size();

  // Argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  INTERNAL_CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Unary operations preserve shape and data type
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_);
}

// ============================================================================
// Registration Function for Block Unary Operations
// ============================================================================

// Helper lambda factory for unary operations
auto MakeUnaryCodegenCCE(const std::string& isa_name) {
  return [isa_name](const CallPtr& op, codegen::CCECodegen& codegen) -> std::string {
    std::string target_var = codegen.GetCurrentResultTarget();
    std::string input = codegen.GetExprAsCode(op->args_[0]);
    codegen.Emit(isa_name + "(" + target_var + ", " + input + ");");
    return target_var;
  };
}

REGISTER_OP("block.neg")
    .set_op_category("BlockOp")
    .set_description("Negation of a tile (element-wise)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.neg");
    })
    .f_codegen_cce(MakeUnaryCodegenCCE("TNEG"));

REGISTER_OP("block.exp")
    .set_op_category("BlockOp")
    .set_description("Exponential function of a tile (element-wise)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.exp");
    })
    .f_codegen_cce(MakeUnaryCodegenCCE("TEXP"));

REGISTER_OP("block.recip")
    .set_op_category("BlockOp")
    .set_description("Reciprocal (1/x) of a tile (element-wise)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.recip");
    })
    .f_codegen_cce(MakeUnaryCodegenCCE("TRECIP"));

REGISTER_OP("block.sqrt")
    .set_op_category("BlockOp")
    .set_description("Square root of a tile (element-wise)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.sqrt");
    })
    .f_codegen_cce(MakeUnaryCodegenCCE("TSQRT"));

REGISTER_OP("block.rsqrt")
    .set_op_category("BlockOp")
    .set_description("Reciprocal square root (1/sqrt(x)) of a tile (element-wise)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.rsqrt");
    })
    .f_codegen_cce(MakeUnaryCodegenCCE("TRSQRT"));

}  // namespace ir
}  // namespace pypto
