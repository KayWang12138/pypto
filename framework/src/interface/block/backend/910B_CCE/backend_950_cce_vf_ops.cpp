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
 * @file backend_950_cce_vf_ops.cpp
 * @brief CCE backend op registration for VF API operations (A5 target).
 *
 * VF ops directly emit VF instructions (vlds, vmax, vdup, etc.) without
 * going through the PTO-ISA intermediate layer. API naming references AscendC.
 */

#include <string>

#include "block/backend/910B_CCE/backend_910b_cce.h"
#include "block/backend/common/backend.h"
#include "block/codegen/cce/cce_codegen.h"
#include "block/codegen/codegen_base.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/pipe.h"
#include "ir/type.h"

namespace pypto {
namespace backend {
using ir::DataType;

// ============================================================================
// Scope markers
// ============================================================================

static std::string EmitVFScopeEnter(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto name = op->GetKwarg<std::string>("name");
  codegen.Emit("// === VF Scope Enter: " + name + " ===");
  codegen.Emit("__VEC_SCOPE__");
  codegen.Emit("{");
  codegen.set_in_vf_scope(true);
  return "";
}

static std::string EmitVFScopeExit(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto name = op->GetKwarg<std::string>("name");
  codegen.set_in_vf_scope(false);
  codegen.Emit("}");
  codegen.Emit("// === VF Scope Exit: " + name + " ===");
  return "";
}

// ============================================================================
// RegTensor declaration
// ============================================================================

static std::string EmitVFRegTensor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto dtype = op->GetKwarg<DataType>("dtype");
  std::string reg_name = codegen.GetCurrentResultTarget();

  std::string type_str = "float";
  if (dtype == DataType::FP32) {
    type_str = "float";
  } else if (dtype == DataType::FP16) {
    type_str = "half";
  } else if (dtype == DataType::BF16) {
    type_str = "bfloat16_t";
  } else if (dtype == DataType::INT32) {
    type_str = "int32_t";
  } else if (dtype == DataType::UINT8) {
    type_str = "uint8_t";
  }

  codegen.Emit("RegTensor<" + type_str + "> " + reg_name + ";");
  return "";
}

// ============================================================================
// CreateMask — declares MaskReg + emits VF init instruction
// ============================================================================

static std::string EmitVFCreateMask(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto pattern = op->GetKwarg<std::string>("pattern");
  auto dtype = op->GetKwarg<DataType>("dtype");
  std::string reg_name = codegen.GetCurrentResultTarget();

  codegen.Emit("MaskReg " + reg_name + ";");

  // Select pset instruction based on data element size (not mask type)
  // float/int32 (4 bytes) → pset_b32, half/bf16 (2 bytes) → pset_b16, int8 (1 byte) → pset_b8
  if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
    codegen.Emit(reg_name + " = pset_b8(PAT_ALL);");
  } else if (dtype == DataType::FP32 || dtype == DataType::INT32 || dtype == DataType::UINT32) {
    codegen.Emit(reg_name + " = pset_b32(PAT_ALL);");
  } else {
    // FP16, BF16, UINT16, INT16 etc. (2 bytes)
    codegen.Emit(reg_name + " = pset_b16(PAT_ALL);");
  }

  return "";
}

// ============================================================================
// Duplicate — scalar broadcast
// ============================================================================

static std::string EmitVFDuplicate(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  // args: [dst, scalar, (optional) mask]
  std::string dst = codegen.GetExprAsCode(op->args_[0]);
  std::string scalar_str = codegen.GetExprAsCode(op->args_[1]);

  if (op->args_.size() >= 3) {
    // With mask: vdup(dst, scalar, preg, MODE_ZEROING)
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    codegen.Emit("vdup(" + dst + ", " + scalar_str + ", " + mask + ", MODE_ZEROING);");
  } else {
    // Without mask: vbr(dst, scalar)
    codegen.Emit("vbr(" + dst + ", " + scalar_str + ");");
  }

  return "";
}

// ============================================================================
// Helper: get __ubuf__ pointer from tile or TileOffsetExpr
// ============================================================================

static std::string GetUBufPtr(codegen::CCECodegen& codegen, const ir::ExprPtr& expr,
                              const std::string& cast_type = "float") {
  std::string code = codegen.GetExprAsCode(expr);
  // Already a __ubuf__ pointer (VF ptr variable or inline TileOffsetExpr)
  if (codegen.IsVFPtrVar(code) || code.find("__ubuf__") != std::string::npos) {
    return "(__ubuf__ " + cast_type + " *)" + code;
  }
  // Plain Tile: need .data()
  return "(__ubuf__ " + cast_type + " *)" + code + ".data()";
}

// ============================================================================
// LoadAlign — vlds / plds
// ============================================================================

static std::string EmitVFLoadAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  // args: [dst, ub_ptr, offset]
  std::string dst_reg = codegen.GetExprAsCode(op->args_[0]);
  std::string offset_str = codegen.GetExprAsCode(op->args_[2]);

  // Get dist mode with default NORM
  std::string dist = "NORM";
  if (op->HasKwarg("dist")) {
    dist = op->GetKwarg<std::string>("dist");
  }

  if (dist == "DS") {
    std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
    codegen.Emit("plds(" + dst_reg + ", " + ub_ptr + ", " + offset_str + ", DS);");
  } else {
    std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], "float");
    codegen.Emit("vlds(" + dst_reg + ", " + ub_ptr + ", " + offset_str + ", " + dist + ");");
  }
  return "";
}

// ============================================================================
// StoreAlign — vsts
// ============================================================================

static std::string EmitVFStoreAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  // args: [dst_ptr, src_reg, mask, (optional) block_stride, (optional) repeat_stride]
  std::string src_reg = codegen.GetExprAsCode(op->args_[1]);
  std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);

  // Get kwargs with defaults
  std::string dist = "NORM_B32";
  if (op->HasKwarg("dist")) {
    dist = op->GetKwarg<std::string>("dist");
  }
  bool post_update = false;
  if (op->HasKwarg("post_update")) {
    post_update = op->GetKwarg<bool>("post_update");
  }
  std::string data_copy_mode = "NORM";
  if (op->HasKwarg("data_copy_mode")) {
    data_copy_mode = op->GetKwarg<std::string>("data_copy_mode");
  }

  // Determine pointer cast type from tile dtype
  std::string ptr_type = "float";
  auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  if (tile_type) {
    if (tile_type->dtype_ == DataType::FP16 || tile_type->dtype_ == DataType::BF16) {
      ptr_type = "half";
    } else if (tile_type->dtype_ == DataType::UINT8 || tile_type->dtype_ == DataType::INT8) {
      ptr_type = "uint8_t";
    }
  }

  if (data_copy_mode == "DATA_BLOCK_COPY") {
    std::string block_stride = "0";
    std::string repeat_stride = "0";
    if (op->args_.size() >= 4) {
      block_stride = codegen.GetExprAsCode(op->args_[3]);
    } else if (op->HasKwarg("block_stride")) {
      block_stride = std::to_string(op->GetKwarg<int>("block_stride"));
    }
    if (op->args_.size() >= 5) {
      repeat_stride = codegen.GetExprAsCode(op->args_[4]);
    } else if (op->HasKwarg("repeat_stride")) {
      repeat_stride = std::to_string(op->GetKwarg<int>("repeat_stride"));
    }
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
    if (post_update) {
      // POST_UPDATE needs pointer reference — reuse if already declared for this tile
      std::string ptr_var = codegen.GetOrCreatePostUpdatePtr(dst_ptr, ptr_type, dst_ptr);
      codegen.Emit("vsstb(" + src_reg + ", " + ptr_var + ", " +
                   "(" + block_stride + " << 16u) | (" + repeat_stride + " & 0xFFFFU), " +
                   mask_reg + ", POST_UPDATE);");
    } else {
      codegen.Emit("vsstb(" + src_reg + ", " + dst_ptr + ", " +
                   "(" + block_stride + " << 16u) | (" + repeat_stride + " & 0xFFFFU), " +
                   mask_reg + ");");
    }
  } else if (post_update) {
    std::string stride = (op->args_.size() >= 4) ? codegen.GetExprAsCode(op->args_[3]) : "0";
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
    std::string ptr_var = "_vf_st_ptr_" + std::to_string(codegen.GetTileOffsetCounter());
    codegen.Emit("__ubuf__ " + ptr_type + " *" + ptr_var + " = " + dst_ptr + ";");
    codegen.Emit("vsts(" + src_reg + ", " + ptr_var + ", " +
                 stride + ", " + dist + ", " + mask_reg + ", POST_UPDATE);");
  } else {
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
    codegen.Emit("vsts(" + src_reg + ", " + dst_ptr + ", 0, " +
                 dist + ", " + mask_reg + ");");
  }
  return "";
}

// ============================================================================
// MemBar — mem_bar
// ============================================================================

static std::string EmitVFMemBar(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string mode = "VST_VLD";
  if (op->HasKwarg("mode")) {
    mode = op->GetKwarg<std::string>("mode");
  }
  codegen.Emit("mem_bar(" + mode + ");");
  return "";
}

// ============================================================================
// Max — vmax
// ============================================================================

static std::string EmitVFMax(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  // Parser args order: [dst, src0, src1, mask]
  CHECK(op->args_.size() == 4) << "vf.Max requires 4 args (dst, src0, src1, mask)";
  std::string dst = codegen.GetExprAsCode(op->args_[0]);
  std::string src0 = codegen.GetExprAsCode(op->args_[1]);
  std::string src1 = codegen.GetExprAsCode(op->args_[2]);
  std::string mask = codegen.GetExprAsCode(op->args_[3]);

  codegen.Emit("vmax(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", MODE_ZEROING);");
  return "";
}

// ============================================================================
// Add — vadd
// ============================================================================

static std::string EmitVFAdd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "vf.Add requires 4 args (dst, src0, src1, mask)";
  std::string dst = codegen.GetExprAsCode(op->args_[0]);
  std::string src0 = codegen.GetExprAsCode(op->args_[1]);
  std::string src1 = codegen.GetExprAsCode(op->args_[2]);
  std::string mask = codegen.GetExprAsCode(op->args_[3]);

  codegen.Emit("vadd(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", MODE_ZEROING);");
  return "";
}

// ============================================================================
// Sub — vsub
// ============================================================================

static std::string EmitVFSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "vf.Sub requires 4 args (dst, src0, src1, mask)";
  std::string dst = codegen.GetExprAsCode(op->args_[0]);
  std::string src0 = codegen.GetExprAsCode(op->args_[1]);
  std::string src1 = codegen.GetExprAsCode(op->args_[2]);
  std::string mask = codegen.GetExprAsCode(op->args_[3]);

  codegen.Emit("vsub(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", MODE_ZEROING);");
  return "";
}

// ============================================================================
// Mul — vmul
// ============================================================================

static std::string EmitVFMul(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "vf.Mul requires 4 args (dst, src0, src1, mask)";
  std::string dst = codegen.GetExprAsCode(op->args_[0]);
  std::string src0 = codegen.GetExprAsCode(op->args_[1]);
  std::string src1 = codegen.GetExprAsCode(op->args_[2]);
  std::string mask = codegen.GetExprAsCode(op->args_[3]);

  codegen.Emit("vmul(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", MODE_ZEROING);");
  return "";
}

// ============================================================================
// Muls — vmuls
// ============================================================================

static std::string EmitVFMuls(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  // Parser args order: [dst, src, scalar, mask]
  CHECK(op->args_.size() == 4) << "vf.Muls requires 4 args (dst, src, scalar, mask)";
  std::string dst = codegen.GetExprAsCode(op->args_[0]);
  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
  std::string mask = codegen.GetExprAsCode(op->args_[3]);

  codegen.Emit("vmuls(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ");");
  return "";
}

// ============================================================================
// Ln — vln
// ============================================================================

static std::string EmitVFLn(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  // Parser args order: [dst, src, mask]
  CHECK(op->args_.size() == 3) << "vf.Ln requires 3 args (dst, src, mask)";
  std::string dst = codegen.GetExprAsCode(op->args_[0]);
  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string mask = codegen.GetExprAsCode(op->args_[2]);

  codegen.Emit("vln(" + dst + ", " + src + ", " + mask + ", MODE_ZEROING);");
  return "";
}

// ============================================================================
// FusedExpSub — vexpdif
// ============================================================================

static std::string EmitVFFusedExpSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  // Parser args order: [dst, src, max, mask]
  CHECK(op->args_.size() == 4) << "vf.FusedExpSub requires 4 args (dst, src, max, mask)";
  std::string dst = codegen.GetExprAsCode(op->args_[0]);
  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string max_reg = codegen.GetExprAsCode(op->args_[2]);
  std::string mask = codegen.GetExprAsCode(op->args_[3]);

  codegen.Emit("vexpdif(" + dst + ", " + src + ", " + max_reg + ", " + mask + ", PART_EVEN);");
  return "";
}

// ============================================================================
// Cast — vcvt
// ============================================================================

static std::string EmitVFCast(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  // Parser args order: [dst, src, mask]
  CHECK(op->args_.size() == 3) << "vf.Cast requires 3 args (dst, src, mask)";
  std::string dst = codegen.GetExprAsCode(op->args_[0]);
  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string mask = codegen.GetExprAsCode(op->args_[2]);

  // Get layout and round_mode with defaults
  std::string layout = "ZERO";
  if (op->HasKwarg("layout")) {
    layout = op->GetKwarg<std::string>("layout");
  }
  std::string round_mode = "CAST_ROUND";
  if (op->HasKwarg("round_mode")) {
    round_mode = op->GetKwarg<std::string>("round_mode");
  }

  // Map layout to Part mode: ZERO→PART_EVEN, ONE→PART_ODD, TWO→PART_TWO, THREE→PART_THREE
  std::string part;
  if (layout == "ZERO") part = "PART_EVEN";
  else if (layout == "ONE") part = "PART_ODD";
  else if (layout == "TWO") part = "PART_TWO";
  else if (layout == "THREE") part = "PART_THREE";
  else part = "PART_EVEN";

  // Map round_mode: CAST_ROUND→ROUND_R, CAST_RINT→ROUND_N
  std::string round;
  if (round_mode == "CAST_RINT") round = "ROUND_N";
  else round = "ROUND_R";

  // fp32→fp16 narrowing: vcvt(dst, src, mask, ROUND_R, RS_DISABLE, PART_EVEN)
  // fp16→fp32 widening:  vcvt(dst, src, mask, PART_EVEN)
  codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", RS_DISABLE, " + part + ");");
  return "";
}

// ============================================================================
// DeInterleave — vdintlv
// ============================================================================

static std::string EmitVFDeInterleave(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  // Parser args order: [dst0, dst1, src0, src1]
  CHECK(op->args_.size() == 4) << "vf.DeInterleave requires 4 args (dst0, dst1, src0, src1)";
  std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
  std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
  std::string src0 = codegen.GetExprAsCode(op->args_[2]);
  std::string src1 = codegen.GetExprAsCode(op->args_[3]);

  codegen.Emit("vdintlv(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
  return "";
}

// ============================================================================
// Select — vsel
// ============================================================================

static std::string EmitVFSelect(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  // Parser args order: [dst, src_true, src_false, mask]
  CHECK(op->args_.size() == 4) << "vf.Select requires 4 args (dst, src_true, src_false, mask)";
  std::string dst = codegen.GetExprAsCode(op->args_[0]);
  std::string src_true = codegen.GetExprAsCode(op->args_[1]);
  std::string src_false = codegen.GetExprAsCode(op->args_[2]);
  std::string mask = codegen.GetExprAsCode(op->args_[3]);

  codegen.Emit("vsel(" + dst + ", " + src_true + ", " + src_false + ", " + mask + ");");
  return "";
}

// ============================================================================
// UpdateMask — plt_b32/plt_b16
// ============================================================================

static std::string EmitVFUpdateMask(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string scalar = codegen.GetExprAsCode(op->args_[0]);
  std::string reg_name = codegen.GetCurrentResultTarget();

  // Default to b32 (float), use dtype kwarg to select b16
  bool use_b16 = false;
  if (op->HasKwarg("dtype")) {
    auto dtype = op->GetKwarg<DataType>("dtype");
    use_b16 = (dtype == DataType::FP16 || dtype == DataType::BF16 ||
               dtype == DataType::UINT16 || dtype == DataType::INT16);
  }

  // plt_b32/plt_b16 requires uint32_t& reference, so declare a variable first
  std::string scalar_var = "_vf_mask_scalar_" + std::to_string(codegen.GetTileOffsetCounter());
  codegen.Emit("uint32_t " + scalar_var + " = (uint32_t)" + scalar + ";");
  codegen.Emit("MaskReg " + reg_name + ";");
  if (use_b16) {
    codegen.Emit(reg_name + " = plt_b16(" + scalar_var + ", POST_UPDATE);");
  } else {
    codegen.Emit(reg_name + " = plt_b32(" + scalar_var + ", POST_UPDATE);");
  }
  return "";
}

// ============================================================================
// Registration
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.vf_scope_enter")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFScopeEnter(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.vf_scope_exit")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFScopeExit(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.RegTensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFRegTensor(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.CreateMask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFCreateMask(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Duplicate")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFDuplicate(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.LoadAlign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFLoadAlign(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.StoreAlign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFStoreAlign(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFMax(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFAdd(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFSub(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFMul(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFMuls(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Ln")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFLn(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.FusedExpSub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFFusedExpSub(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFCast(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.DeInterleave")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFDeInterleave(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.Select")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFSelect(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.UpdateMask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFUpdateMask(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "vf.MemBar")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return EmitVFMemBar(op, codegen);
    });

}  // namespace backend
}  // namespace pypto
