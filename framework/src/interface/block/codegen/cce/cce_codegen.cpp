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

#include "block/codegen/cce/cce_codegen.h"

#include <cctype>
#include <cstddef>
#include <functional>
#include <ios>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "block/backend/common/backend.h"
#include "block/backend/common/backend_config.h"
#include "block/codegen/orchestration/orchestration_codegen.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"

namespace pypto {
namespace codegen {
using ir::DataType;

const char KERNEL_HEADER[] = R"(
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif
)";

// Header for single-file mode (no tensor.h needed �?uses direct __gm__ pointers)
const char KERNEL_HEADER_SINGLE[] = R"(
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;
)";

namespace {

bool IsNZTensorType(const ir::TensorTypePtr& tensor_type) {
  return tensor_type && tensor_type->tensor_view_.has_value() &&
         tensor_type->tensor_view_->layout == ir::TensorLayout::NZ;
}

bool IsIdentStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool IsIdentChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

std::string TrimCopy(const std::string& s) {
  size_t begin = 0;
  while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
    ++begin;
  }
  size_t end = s.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
    --end;
  }
  return s.substr(begin, end - begin);
}

bool IsWritableLValueExpr(const std::string& expr) {
  std::string s = TrimCopy(expr);
  if (s.empty()) return false;

  size_t i = 0;
  if (!IsIdentStart(s[i])) return false;
  while (i < s.size() && IsIdentChar(s[i])) {
    ++i;
  }

  while (i < s.size()) {
    if (s[i] == '.') {
      ++i;
      if (i >= s.size() || !IsIdentStart(s[i])) return false;
      while (i < s.size() && IsIdentChar(s[i])) {
        ++i;
      }
      continue;
    }

    if (s[i] == '-' && i + 1 < s.size() && s[i + 1] == '>') {
      i += 2;
      if (i >= s.size() || !IsIdentStart(s[i])) return false;
      while (i < s.size() && IsIdentChar(s[i])) {
        ++i;
      }
      continue;
    }

    if (s[i] == '[') {
      int depth = 1;
      ++i;
      while (i < s.size() && depth > 0) {
        if (s[i] == '[') {
          ++depth;
        } else if (s[i] == ']') {
          --depth;
        }
        ++i;
      }
      if (depth != 0) return false;
      continue;
    }

    return false;
  }

  return true;
}

int64_t GetNZInnerCols(const DataType& dtype) {
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
  throw pypto::ir::ValueError("CCE NZ tensor lowering does not support dtype " + dtype.ToString());
}

void ValidateStaticNZTensorShape(const ir::TensorTypePtr& tensor_type,
                                 const std::vector<int64_t>& logical_dims) {
  CHECK(tensor_type != nullptr) << "CCE NZ tensor lowering requires a valid TensorType";
  if (logical_dims.size() != 2) {
    throw pypto::ir::ValueError("CCE NZ tensor lowering currently requires a 2D tensor");
  }

  const int64_t rows = logical_dims[0];
  const int64_t cols = logical_dims[1];
  const int64_t c0 = GetNZInnerCols(tensor_type->dtype_);
  if (rows % 16 != 0 || cols % c0 != 0) {
    throw pypto::ir::ValueError(
        "CCE NZ tensor lowering requires rows divisible by 16 and cols divisible by the destination C0 size");
  }
}

std::vector<int64_t> BuildNZPhysicalShapeDims(const ir::TensorTypePtr& tensor_type,
                                              const std::vector<int64_t>& logical_dims) {
  ValidateStaticNZTensorShape(tensor_type, logical_dims);
  const int64_t c0 = GetNZInnerCols(tensor_type->dtype_);
  return {1, logical_dims[1] / c0, logical_dims[0] / 16, 16, c0};
}

}  // namespace

CCECodegen::CCECodegen() : backend_(backend::GetBackend()) {
  auto type = backend::GetBackendType();
  CHECK(type == backend::BackendType::CCE)
      << "CCECodegen requires CCE backend, but " << (type == backend::BackendType::PTO ? "PTO" : "unknown")
      << " is configured";
}

std::map<std::string, std::string> CCECodegen::Generate(const ir::ProgramPtr& program) {
  CHECK(program != nullptr) << "Cannot generate code for null program";

  std::map<std::string, std::string> files;

  // Separate functions into kernel functions and orchestration functions
  std::vector<ir::FunctionPtr> kernel_functions;
  std::vector<ir::FunctionPtr> orchestration_functions;

  for (const auto& [gvar, func] : program->functions_) {
    if (func->funcType_ == ir::FunctionType::ORCHESTRATION) {
      orchestration_functions.push_back(func);
    } else {
      kernel_functions.push_back(func);
    }
  }

  // Generate kernel functions (CCE C++ code) to separate .cpp files in kernels/ subdirectory
  for (const auto& func : kernel_functions) {
    std::ostringstream oss;
    oss << "// Kernel Function: " << func->name_ << "\n";
    oss << "// Generated by PyPTO IR Compiler\n\n";
    oss << KERNEL_HEADER << "\n";

    std::string func_code = GenerateFunction(func);
    oss << func_code;

    auto core_type = InferFunctionCoreType(func);
    std::string core_type_str = core_type == ir::CoreType::VECTOR ? "aiv" : "aic";

    std::string filename = "kernels/" + core_type_str + "/" + func->name_ + ".cpp";
    files[filename] = oss.str();
  }

  // Generate orchestration function to orchestration/ subdirectory
  if (!orchestration_functions.empty()) {
    if (orchestration_functions.size() > 1) {
      throw pypto::ir::ValueError("Program should have exactly one orchestration function, got " +
                              std::to_string(orchestration_functions.size()));
    }

    const auto& func = orchestration_functions[0];
    std::ostringstream oss;
    oss << "// Orchestration Function: " << func->name_ << "\n";
    oss << "// Generated by PyPTO IR Compiler\n\n";

    auto orch_result = GenerateOrchestration(program, func);
    oss << orch_result.code;

    std::string filename = "orchestration/" + func->name_ + ".cpp";
    files[filename] = oss.str();

    // Generate config file
    std::string config_code =
        GenerateConfigFile(func->name_, orch_result.func_name_to_id, orch_result.func_name_to_core_type);
    files["kernel_config.py"] = config_code;
  }

  return files;
}

std::string CCECodegen::GenerateConfigFile(
    const std::string& orch_func_name, const std::map<std::string, int>& func_name_to_id,
    const std::map<std::string, ir::CoreType>& func_name_to_core_type) {
  std::ostringstream oss;
  oss << "# Kernel and Orchestration Configuration\n\n";
  oss << "from pathlib import Path\n\n";
  oss << "_ROOT_DIR = Path(__file__).parent\n\n";

  oss << "# Runtime configuration for tensormap_and_ringbuffer\n";
  oss << "# This runtime requires 4 AICPU threads (3 schedulers + 1 orchestrator on thread 3)\n";
  oss << "RUNTIME_CONFIG = {\n";
  oss << "\t\"runtime\": \"tensormap_and_ringbuffer\",\n";
  oss << "\t\"aicpu_thread_num\": 4,\n";
  oss << "\t\"block_dim\": 24,\n";
  oss << "}\n\n";

  oss << "ORCHESTRATION = {\n\t\"source\": str(_ROOT_DIR / \"orchestration\" / \"" << orch_func_name
      << ".cpp\"),\n"
      << "\t\"function_name\": \"aicpu_orchestration_entry\"\n}\n\n";

  oss << "KERNELS = [\n";
  for (const auto& [name, id] : func_name_to_id) {
    std::string core_type = func_name_to_core_type.at(name) == ir::CoreType::VECTOR ? "aiv" : "aic";
    oss << "\t{\"func_id\": " << id << R"(, "source": str(_ROOT_DIR / "kernels" / ")" << core_type
        << "\" / \"" << name << R"(.cpp"), "core_type": ")" << core_type << "\"},\n";
  }
  oss << "]\n";
  return oss.str();
}

std::string CCECodegen::GenerateFunction(const ir::FunctionPtr& func) {
  CHECK(func != nullptr) << "Cannot generate code for null function";

  // Clear state
  emitter_.Clear();
  context_.Clear();

  // Generate prologue and body
  GeneratePrologue(func);
  GenerateBody(func);

  return emitter_.GetCode();
}

// ============================================================================
// Single-file MIX mode generation (skip ptoas)
// ============================================================================

std::string CCECodegen::GenerateSingle(const ir::ProgramPtr& program, const std::string& arch) {
  CHECK(program != nullptr) << "Cannot generate code for null program";

  // Store arch for use in prologue generation
  arch_ = arch;

  // Run IR passes (same as PTOCodegen::Generate)
  ir::ProgramPtr lowered = ir::pass::LowerBreakContinue()(program);
  ir::ProgramPtr ssa_program = ir::pass::ConvertToSSA()(lowered);
  ir::ProgramPtr opt_program = ir::pass::ConstFoldAndSimplify()(ssa_program);

  // Find the kernel function (non-orchestration)
  ir::FunctionPtr kernel_func;
  for (const auto& [gvar, func] : opt_program->functions_) {
    if (func->funcType_ != ir::FunctionType::ORCHESTRATION) {
      kernel_func = func;
      break;
    }
  }
  CHECK(kernel_func != nullptr) << "No kernel function found in program";

  // Clear state
  emitter_.Clear();
  context_.Clear();
  single_file_mode_ = true;

  // Pre-scan: collect Var names used as runtime buf_id (Mutex) so the N-way
  // optimizer can emit a uint8_t/_bid_ array for them instead of event_t/_eid_.
  buf_id_var_names_.clear();
  CollectBufIdVarNames(kernel_func->body_, buf_id_var_names_);

  // Pre-scan: collect Var names that appear in any read position, used to
  // drop unused phi return_vars in IfStmt codegen.
  var_read_names_.clear();
  CollectVarReadNames(kernel_func->body_, var_read_names_);

  // Pre-scan: collect mutex_id → pipe mappings per section for A5 V-pipe optimization.
  cube_mutex_pipes_.clear();
  vec_mutex_pipes_.clear();
  CollectMutexPipeInfo(kernel_func->body_);

  // Detect cross-core sync (a5 uses hardware sync, not ffts)
  bool has_cross_sync = DetectCrossCoreSyncOps(kernel_func->body_);
  bool needs_ffts = has_cross_sync && (arch_ != "a5");

  // Emit header (single-file mode: no tensor.h)
  emitter_.EmitLine(KERNEL_HEADER_SINGLE);

  // Pre-scan: collect and emit struct type definitions before the function
  struct_type_defs_.clear();
  PreEmitStructTypes(kernel_func->body_);
  if (!struct_type_defs_.empty()) {
    emitter_.EmitLine("");
  }

  // Emit BufferSlot template struct for NBuffer tile+bid arrays
  emitter_.EmitLine("template <typename TileData>");
  emitter_.EmitLine("struct BufferSlot { TileData tile; uint8_t bid; };");
  emitter_.EmitLine("");
  buffer_slot_struct_emitted_ = true;
  buffer_slot_decls_emitted_.clear();

  // Generate prologue and body
  GenerateSinglePrologue(kernel_func, needs_ffts);
  section_snapshot_saved_ = false;
  GenerateBody(kernel_func);

  single_file_mode_ = false;
  return emitter_.GetCode();
}

namespace {
/**
 * @brief Collect tile variable names that are actually referenced (used) in ops.
 *
 * Walks the IR looking for Var/IterArg references with TileType in Call arguments.
 * Variables that are only declared (via AssignStmt) but never used as operands
 * are excluded. This filters out redundant prologue tiles (_tuple_tmp_*, *_buf_*).
 */
class TileUsageCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitStmt_;
    using ir::IRVisitor::VisitExpr_;
 public:
  // Tiles used as direct operands in Call ops (TLOAD, TMOV, etc.) or IfStmt yields
  std::set<std::string> op_used_names_;
  // Tiles used as elements in tuple construction (MakeTuple) �?only grouping
  std::set<std::string> tuple_only_names_;
  // Map from Call expr pointer to the Var that holds its result (from AssignStmt)
  std::map<const ir::Call*, ir::VarPtr> call_to_var_;

  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    // Build mapping from Call value to the Var that holds it
    if (auto call = ir::As<ir::Call>(op->value_)) {
      call_to_var_[call.get()] = op->var_;
    }
    // Record let-binding edges `lhs = rhs_var` for tile-typed vars so callers can
    // propagate `ifstmt_return_var_names_` across copy-assignments. Needed because
    // `let_x = _tidx_N` inherits _tidx_N's TileType (and its memref address from
    // the first yield branch), which would otherwise cause spurious dedup aliases
    // like `auto& let_x = <first_branch_tile>;` even though let_x's real value is
    // a dynamic array[idx] produced by the N-way-select optimization.
    if (op->var_ && op->value_ &&
        (ir::As<ir::TileType>(op->var_->GetType()) ||
         ir::As<ir::TupleType>(op->var_->GetType()))) {
      if (auto rhs_var = ir::As<ir::Var>(op->value_)) {
        tile_assign_edges_.emplace_back(op->var_->name_, rhs_var->name_);
      }
    }
    ir::IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const ir::CallPtr& op) override {
    // Tile vars in Call args are truly used by hardware operations
    for (const auto& arg : op->args_) {
      CollectTileNames(arg, true);
    }
    ir::IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ir::IterArgPtr& op) override {
    if (ir::As<ir::TileType>(op->GetType()) || ir::As<ir::TupleType>(op->GetType())) {
      op_used_names_.insert(op->name_);
    }
    if (op->initValue_) VisitExpr(op->initValue_);
  }

  void VisitStmt_(const ir::YieldStmtPtr& op) override {
    // Tiles in yield values are used (selected by IfStmt/ForStmt)
    for (const auto& val : op->value_) {
      CollectTileNames(val, true);
    }
  }

  // Collect IfStmt return var names (these will be handled by IfStmt optimization,
  // not the prologue)
  void VisitStmt_(const ir::IfStmtPtr& op) override {
    for (const auto& rv : op->returnVars_) {
      if (ir::As<ir::TileType>(rv->GetType())) {
        ifstmt_return_var_names_.insert(rv->name_);
      }
    }
    ir::IRVisitor::VisitStmt_(op);
  }

  std::set<std::string> ifstmt_return_var_names_;
  // (lhs, rhs) pairs for tile-typed `lhs = rhs_var` copy-assignments.
  std::vector<std::pair<std::string, std::string>> tile_assign_edges_;

  // Propagate ifstmt_return_var_names_ across tile copy-assignments to a
  // fixpoint. If `a = _tidx_N` and `b = a`, both a and b are marked so they
  // skip prologue emission / dedup aliasing.
  void PropagateIfStmtReturnVars() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto& [lhs, rhs] : tile_assign_edges_) {
        if (ifstmt_return_var_names_.count(rhs) &&
            !ifstmt_return_var_names_.count(lhs)) {
          ifstmt_return_var_names_.insert(lhs);
          changed = true;
        }
      }
    }
  }

  // Get names of tiles that are used in ops or yields (not just tuple grouping)
  std::set<std::string> GetUsedNames() const { return op_used_names_; }

 private:
  void CollectTileNames(const ir::ExprPtr& expr, bool is_op_arg) {
    if (!expr) return;
    if (auto var = ir::As<ir::Var>(expr)) {
      if (ir::As<ir::TileType>(var->GetType()) || ir::As<ir::TupleType>(var->GetType())) {
        if (is_op_arg) {
          op_used_names_.insert(var->name_);
        } else {
          tuple_only_names_.insert(var->name_);
        }
      }
    } else if (auto iter_arg = ir::As<ir::IterArg>(expr)) {
      if (ir::As<ir::TileType>(iter_arg->GetType()) || ir::As<ir::TupleType>(iter_arg->GetType())) {
        op_used_names_.insert(iter_arg->name_);
      }
    } else if (auto call = ir::As<ir::Call>(expr)) {
      // Check if this Call returns a TileType and is assigned to a Var
      if (ir::As<ir::TileType>(call->GetType()) || ir::As<ir::TupleType>(call->GetType())) {
        auto it = call_to_var_.find(call.get());
        if (it != call_to_var_.end()) {
          if (is_op_arg) {
            op_used_names_.insert(it->second->name_);
          } else {
            tuple_only_names_.insert(it->second->name_);
          }
        }
      }
      // Also recurse into Call args
      for (const auto& arg : call->args_) {
        CollectTileNames(arg, is_op_arg);
      }
    } else if (auto tge = ir::As<ir::TupleGetItemExpr>(expr)) {
      // TupleGetItem accesses a tuple element �?the tuple itself is used
      CollectTileNames(tge->tuple_, is_op_arg);
    } else if (auto toe = ir::As<ir::TileOffsetExpr>(expr)) {
      // TileOffsetExpr accesses a tile with element offset �?the base tile is used
      CollectTileNames(toe->tile_, is_op_arg);
    } else if (auto mt = ir::As<ir::MakeTuple>(expr)) {
      // MakeTuple elements are just grouping �?NOT direct operands
      for (const auto& elem : mt->elements_) {
        CollectTileNames(elem, false);
      }
    }
  }
};

/**
 * @brief Collect which sections each tile variable is USED in (as Call operand).
 *
 * A tile used in both Cube and Vec sections (e.g., written by CUBE, read by VEC)
 * should be declared as shared (outside any #if guard).
 */
class TileUsageSectionCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitStmt_;
    using ir::IRVisitor::VisitExpr_;
 public:
  std::map<std::string, std::set<ir::SectionKind>> tile_usage_sections_;
  std::optional<ir::SectionKind> current_section_;

  // Map tuple Var names → their element tile Var names (from MakeTuple AssignStmt).
  std::map<std::string, std::vector<std::string>> tuple_element_names_;

  void VisitStmt_(const ir::SectionStmtPtr& op) override {
    auto prev = current_section_;
    current_section_ = op->sectionKind_;
    ir::IRVisitor::VisitStmt_(op);
    current_section_ = prev;
  }

  // Collect MakeTuple → element tile Var associations (outside or inside sections).
  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    if (op && op->value_) {
      if (auto mt = ir::As<ir::MakeTuple>(op->value_)) {
        auto var = op->var_;
        if (var && ir::As<ir::TupleType>(var->GetType())) {
          for (const auto& elem : mt->elements_) {
            if (auto elem_var = ir::As<ir::Var>(elem)) {
              if (ir::As<ir::TileType>(elem_var->GetType())) {
                tuple_element_names_[var->name_].push_back(elem_var->name_);
              }
            }
          }
        }
      }
    }
    ir::IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const ir::CallPtr& op) override {
    if (current_section_.has_value()) {
      for (const auto& arg : op->args_) {
        CollectTileVarNames(arg);
      }
    }
    ir::IRVisitor::VisitExpr_(op);
  }

  // YieldStmt args also reference tiles (NBuffer if-chain uses yield for tile selection).
  void VisitStmt_(const ir::YieldStmtPtr& op) override {
    if (current_section_.has_value() && op) {
      for (const auto& val : op->value_) {
        CollectTileVarNames(val);
      }
    }
    ir::IRVisitor::VisitStmt_(op);
  }

  // After traversal, propagate section usage from tuples to their element tiles.
  void PropagateToTupleElements() {
    for (const auto& [tuple_name, elem_names] : tuple_element_names_) {
      auto it = tile_usage_sections_.find(tuple_name);
      if (it == tile_usage_sections_.end()) continue;
      for (const auto& elem_name : elem_names) {
        for (const auto& section : it->second) {
          tile_usage_sections_[elem_name].insert(section);
        }
      }
    }
  }

 private:
  void CollectTileVarNames(const ir::ExprPtr& expr) {
    if (!expr) return;
    if (auto var = ir::As<ir::Var>(expr)) {
      if (ir::As<ir::TileType>(var->GetType())) {
        tile_usage_sections_[var->name_].insert(*current_section_);
      }
      // Also track tuple Var usage (for propagation to elements)
      if (ir::As<ir::TupleType>(var->GetType())) {
        tile_usage_sections_[var->name_].insert(*current_section_);
      }
    } else if (auto tge = ir::As<ir::TupleGetItemExpr>(expr)) {
      CollectTileVarNames(tge->tuple_);
    } else if (auto toe = ir::As<ir::TileOffsetExpr>(expr)) {
      CollectTileVarNames(toe->tile_);
    }
  }
};


// Extract valid_shape constructor arguments from a TileType for CCE code generation.
//
// needs_ctor == true  �?template has -1 params (dynamic), Tile constructor must receive runtime values.
// needs_ctor == false �?template has explicit static params, no constructor args needed.
//
// Mapping for each valid_shape element:
//   absent / ConstInt(-1) �?template param = -1, ctor_arg = rows/cols  (needs_ctor = true)
//   ConstInt(N > 0)       �?template param = N,  ctor_arg unused        (needs_ctor = false)
//   Var(name)             �?template param = -1 (skipped by ConvertTileType), ctor_arg = var_name
struct ValidShapeInfo {
  std::string row_ctor_arg;
  std::string col_ctor_arg;
  bool needs_ctor = false;  // true when template uses -1 and constructor args are required
};

inline ValidShapeInfo ExtractValidShapeInfo(const ir::TileTypePtr& tile_type, int64_t rows, int64_t cols,
                                            std::function<std::string(const ir::VarPtr&)> get_var_name) {
  ValidShapeInfo info;
  if (!tile_type->tile_view_.has_value()) {
    // No tile_view: absent valid_shape �?template uses -1, ctor uses full shape.
    info.row_ctor_arg = std::to_string(rows);
    info.col_ctor_arg = std::to_string(cols);
    info.needs_ctor = true;
    return info;
  }
  const auto& tv = tile_type->tile_view_.value();
  if (tv.valid_shape.empty()) {
    // valid_shape absent: template uses -1, ctor uses full shape.
    info.row_ctor_arg = std::to_string(rows);
    info.col_ctor_arg = std::to_string(cols);
    info.needs_ctor = true;
    return info;
  }
  // valid_shape provided: check whether any element is -1 or a runtime Var.
  auto resolve_dim = [&](const ir::ExprPtr& expr, int64_t fallback, std::string& out_arg) -> bool {
    if (auto var = ir::As<ir::Var>(expr)) {
      out_arg = get_var_name(var);
      return true;  // runtime var �?needs ctor
    }
    if (auto c = ir::As<ir::ConstInt>(expr)) {
      if (c->value_ == -1) {
        out_arg = std::to_string(fallback);
        return true;  // -1 sentinel �?template gets -1, ctor gets actual dim
      }
      out_arg = std::to_string(c->value_);
      return false;  // explicit static value �?no ctor needed
    }
    return false;
  };
  bool row_dynamic = false;
  bool col_dynamic = false;
  if (tv.valid_shape.size() >= 1) row_dynamic = resolve_dim(tv.valid_shape[0], rows, info.row_ctor_arg);
  if (tv.valid_shape.size() >= 2) col_dynamic = resolve_dim(tv.valid_shape[1], cols, info.col_ctor_arg);
  info.needs_ctor = row_dynamic || col_dynamic;
  return info;
}

// Build Tile constructor argument string.
// Only call when ValidShapeInfo::needs_ctor is true.
inline std::string BuildTileCtorArgs(const ValidShapeInfo& vs, int64_t rows, int64_t cols) {
  std::string r = vs.row_ctor_arg.empty() ? std::to_string(rows) : vs.row_ctor_arg;
  std::string c = vs.col_ctor_arg.empty() ? std::to_string(cols) : vs.col_ctor_arg;
  return r + ", " + c;
}

}  // namespace

// ========================================================================
// Phase 6 helpers: GenerateSinglePrologue sub-functions
// ========================================================================

void CCECodegen::EmitSingleFunctionSignature(const ir::FunctionPtr& func, bool has_cross_sync) {
  // Collect dynamic dim variables from tensor shapes (first-occurrence order)
  std::vector<ir::VarPtr> dyn_dim_vars;
  std::set<std::string> seen_dyn_names;
  for (const auto& param : func->params_) {
    if (auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType())) {
      for (const auto& dim_expr : tensor_type->shape_) {
        if (auto dim_var = std::dynamic_pointer_cast<const ir::Var>(dim_expr)) {
          if (seen_dyn_names.insert(dim_var->name_).second) {
            dyn_dim_vars.push_back(dim_var);
          }
        }
      }
    }
  }

  // Build PTO-style function signature: __global__ AICORE void func_name(__gm__ type* p1, ...)
  std::ostringstream sig;
  sig << "__global__ AICORE void " << func->name_ << "(";
  bool first = true;
  for (const auto& param : func->params_) {
    if (!first) sig << ", ";
    first = false;
    if (auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType())) {
      std::string element_type = tensor_type->dtype_.ToCTypeString();
      std::string param_name = context_.SanitizeName(param);
      sig << "__gm__ " << element_type << "* " << param_name;
    } else if (auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(param->GetType())) {
      std::string cpp_type = scalar_type->dtype_.ToCTypeString();
      std::string param_name = context_.SanitizeName(param);
      sig << cpp_type << " " << param_name;
    } else if (auto ptr_type = std::dynamic_pointer_cast<const ir::PtrType>(param->GetType())) {
      std::string element_type = ptr_type->dtype_.ToCTypeString();
      std::string param_name = context_.SanitizeName(param);
      sig << "__gm__ " << element_type << "* " << param_name;
    }
  }
  // Append dynamic dim variables as int64_t parameters
  for (const auto& dyn_var : dyn_dim_vars) {
    if (!first) sig << ", ";
    first = false;
    sig << "int32_t " << context_.SanitizeName(dyn_var);
  }
  if (has_cross_sync) {
    if (!first) sig << ", ";
    sig << "__gm__ int64_t* ffts_addr";
  }
  sig << ")";

  emitter_.EmitLine(sig.str());
  emitter_.EmitLine("{");
  emitter_.IncreaseIndent();

  // Register dynamic dim parameters directly (no _local_ copy needed)
  for (const auto& dyn_var : dyn_dim_vars) {
    std::string param_name = context_.SanitizeName(dyn_var);
    context_.RegisterVar(dyn_var, param_name);
  }

  // Emit set_ffts_base_addr if cross-core sync
  if (has_cross_sync) {
    emitter_.EmitLine("set_ffts_base_addr((unsigned long)ffts_addr);");
  }
  emitter_.EmitLine("");
}

void CCECodegen::EmitSingleGlobalTensors(const ir::FunctionPtr& func,
                                          const SectionAccessShapes& section_shapes) {
  // Classify which sections each tensor parameter is used in
  // Register all tensor parameters first (name mapping only)
  std::vector<std::pair<ir::VarPtr, std::string>> tensor_params;  // (param, global_name)
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    const std::string param_name = context_.SanitizeName(param);

    if (auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType())) {
      const std::string global_name = param_name + "Global";
      context_.RegisterVar(param, global_name);
      tensor_params.emplace_back(param, global_name);
      // In single-file mode, register pointer mapping for all tensor parameters
      // The pointer is the parameter name itself (a __gm__ type* pointer)
      context_.RegisterPointer(global_name, param_name);
    } else if (auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(param->GetType())) {
      context_.RegisterVar(param, param_name);
    }
  }

  // Helper lambda to emit GlobalTensor declarations for a set of tensors
  auto emit_global_tensors = [&](const std::map<std::string, std::vector<ir::ExprPtr>>& shapes,
                                 const std::map<std::string, std::vector<int>>& tile_dims) {
    for (const auto& [param, global_name] : tensor_params) {
      auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType());
      if (!tensor_type) continue;
      auto it = shapes.find(param->name_);
      if (it == shapes.end()) continue;
      const std::string param_name = context_.SanitizeName(param);
      std::optional<std::vector<ir::ExprPtr>> access_shape = it->second;
      force_dn_layout_ = (dn_tensors_.count(param->name_) > 0);
      auto tile_dims_it = tile_dims.find(param->name_);
      current_tile_dims_ = tile_dims_it == tile_dims.end()
                               ? std::optional<std::vector<int>>()
                               : std::optional<std::vector<int>>(tile_dims_it->second);
      GenerateGlobalTensorTypeDeclaration(global_name, tensor_type, param_name, std::nullopt, access_shape);
      force_dn_layout_ = false;
      current_tile_dims_.reset();
      emitter_.EmitLine("");
    }
  };

  // Emit GlobalTensors for tensors only accessed outside any section (common area)
  if (!section_shapes.common_shapes.empty()) {
    emit_global_tensors(section_shapes.common_shapes, section_shapes.common_tile_dims);
  }

  // Emit GlobalTensors for tensors accessed in Cube section (with Cube access shapes)
  if (!section_shapes.cube_shapes.empty()) {
    emitter_.EmitLine("#if defined(__DAV_CUBE__)");
    emit_global_tensors(section_shapes.cube_shapes, section_shapes.cube_tile_dims);
    emitter_.EmitLine("#endif");
    emitter_.EmitLine("");
  }

  // Emit GlobalTensors for tensors accessed in Vector section (with Vector access shapes)
  if (!section_shapes.vec_shapes.empty()) {
    emitter_.EmitLine("#if defined(__DAV_VEC__)");
    emit_global_tensors(section_shapes.vec_shapes, section_shapes.vec_tile_dims);
    emitter_.EmitLine("#endif");
    emitter_.EmitLine("");
  }
}

void CCECodegen::EmitSingleTileDeclarations(const ir::FunctionPtr& func) {
  // Collect tile sections for section-aware declarations
  auto tile_sections = CollectTileSections(func->body_);

  // Collect which sections each tile is used in (for cross-section detection)
  TileUsageSectionCollector usage_section_collector;
  if (func->body_) {
    usage_section_collector.VisitStmt(func->body_);
  }
  // Propagate section usage from tuple Vars to their element tile Vars.
  // NBuffer tile tuples are declared outside sections but referenced inside
  // via TupleGetItemExpr; this ensures elements inherit the correct section.
  usage_section_collector.PropagateToTupleElements();
  const auto& tile_usage_sections = usage_section_collector.tile_usage_sections_;

  // Filter and deduplicate tiles
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> all_tiles;
  std::vector<std::pair<ir::VarPtr, ir::VarPtr>> deduped_aliases;
  auto tile_vars = FilterPrologueTiles(func, all_tiles, deduped_aliases);

  // Emit section-aware tile declarations
  if (!tile_vars.empty()) {
    EmitSectionAwareTiles(tile_vars, all_tiles, deduped_aliases, tile_sections, tile_usage_sections);
  }
}

std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> CCECodegen::FilterPrologueTiles(
    const ir::FunctionPtr& func,
    std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>>& all_tiles_out,
    std::vector<std::pair<ir::VarPtr, ir::VarPtr>>& deduped_aliases_out) {
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> tile_vars;
  if (!func->body_) return tile_vars;

  all_tiles_out = CollectTileVariables(func->body_);

  // Collect actually-used tile names
  TileUsageCollector usage_collector;
  usage_collector.VisitStmt(func->body_);
  // Propagate IfStmt return_var-ness across tile copy-assignments so let-bindings
  // like `gmax_p = _tidx_N` (after N-way-select) also skip prologue emission
  // and same-address dedup aliasing.
  usage_collector.PropagateIfStmtReturnVars();
  const auto& used = usage_collector.GetUsedNames();
  const auto& ifstmt_rvs = usage_collector.ifstmt_return_var_names_;

  std::set<std::string> kept_tile_addrs;
  std::map<std::string, ir::VarPtr> kept_tile_addr_vars;

  auto has_independent_runtime_metadata = [](const ir::TileTypePtr& tile_type) {
    return tile_type != nullptr && tile_type->tile_view_.has_value() &&
           !tile_type->tile_view_->valid_shape.empty();
  };

  // Pass 1: Collect FIFO buffer elements
  std::set<std::string> fifo_buf_names;
  for (const auto& [var, tile_type] : all_tiles_out) {
    const std::string& name = var->name_;
    if (used.count(name) == 0) {
      auto last_underscore = name.rfind('_');
      if (last_underscore != std::string::npos) {
        std::string parent = name.substr(0, last_underscore);
        if (used.count(parent) > 0 && parent.find("_tuple_tmp") == std::string::npos) {
          fifo_buf_names.insert(name);
        }
      }
    }
  }

  // Pass 2: Filter tiles
  for (const auto& [var, tile_type] : all_tiles_out) {
    const std::string& name = var->name_;
    if (ifstmt_rvs.count(name) > 0) continue;
    if (used.count(name) == 0 && usage_collector.tuple_only_names_.count(name) > 0) continue;

    bool is_used = used.count(name) > 0 || fifo_buf_names.count(name) > 0;

    // Address+type dedup
    if (is_used && tile_type->memref_.has_value()) {
      // Same-address tiles with explicit valid_shape metadata must stay as
      // distinct C++ objects, otherwise a later SetValidShape() on one tile
      // narrows every deduped alias that shares the same backing storage.
      if (has_independent_runtime_metadata(tile_type)) {
        tile_vars.emplace_back(var, tile_type);
        continue;
      }

      int64_t addr = ExtractConstInt((*tile_type->memref_)->addr_);
      auto space = (*tile_type->memref_)->memory_space_;
      std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);
      std::string type_key = type_converter_.ConvertTileType(tile_type,
          shape_dims.size() >= 1 ? shape_dims[0] : 1, shape_dims.size() >= 2 ? shape_dims[1] : 1);
      std::string dedup_key = std::to_string(static_cast<int>(space)) + ":" +
                              std::to_string(addr) + ":" + type_key;
      if (kept_tile_addrs.count(dedup_key) > 0) {
        auto it = kept_tile_addr_vars.find(dedup_key);
        if (it != kept_tile_addr_vars.end()) {
          deduped_aliases_out.emplace_back(var, it->second);
        }
        continue;
      }
      kept_tile_addrs.insert(dedup_key);
      kept_tile_addr_vars[dedup_key] = var;
    }
    if (is_used) {
      tile_vars.emplace_back(var, tile_type);
    }
  }
  return tile_vars;
}

void CCECodegen::EmitSectionAwareTiles(
    const std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>>& tile_vars,
    const std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>>& all_tiles,
    const std::vector<std::pair<ir::VarPtr, ir::VarPtr>>& deduped_aliases,
    const std::map<ir::VarPtr, ir::SectionKind>& tile_sections,
    const std::map<std::string, std::set<ir::SectionKind>>& tile_usage_sections) {
  // Separate tiles by section
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> cube_tiles;
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> vec_tiles;
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> shared_tiles;

  // Helper: look up parent tuple usage sections for NBuffer tile elements.
  // NBuffer tiles are named like _nbuf_*_tiles_0_0 — strip trailing _N to
  // find the tuple Var name and check if it's used across sections.
  auto find_parent_usage = [&](const std::string& name)
      -> std::map<std::string, std::set<ir::SectionKind>>::const_iterator {
    std::string cur = name;
    auto pos = cur.rfind('_');
    while (pos != std::string::npos && pos > 0) {
      cur = cur.substr(0, pos);
      auto it = tile_usage_sections.find(cur);
      if (it != tile_usage_sections.end()) return it;
      pos = cur.rfind('_');
    }
    return tile_usage_sections.end();
  };

  for (const auto& [var, tile_type] : tile_vars) {
    // Check direct usage across multiple sections → shared
    auto usage_it = tile_usage_sections.find(var->name_);
    if (usage_it != tile_usage_sections.end() && usage_it->second.size() > 1) {
      shared_tiles.emplace_back(var, tile_type);
      continue;
    }
    // Check parent tuple usage (NBuffer tiles)
    if (usage_it == tile_usage_sections.end()) {
      usage_it = find_parent_usage(var->name_);
    }
    if (usage_it != tile_usage_sections.end() && usage_it->second.size() > 1) {
      shared_tiles.emplace_back(var, tile_type);
      continue;
    }
    // Single-section usage from direct or parent lookup
    if (usage_it != tile_usage_sections.end() && usage_it->second.size() == 1) {
      if (*usage_it->second.begin() == ir::SectionKind::Cube) {
        cube_tiles.emplace_back(var, tile_type);
      } else {
        vec_tiles.emplace_back(var, tile_type);
      }
      continue;
    }
    // Fallback: use tile_sections or memory space
    auto it = tile_sections.find(var);
    if (it != tile_sections.end()) {
      if (it->second == ir::SectionKind::Cube) {
        cube_tiles.emplace_back(var, tile_type);
      } else {
        vec_tiles.emplace_back(var, tile_type);
      }
    } else if (tile_type->memref_.has_value()) {
      auto space = (*tile_type->memref_)->memory_space_;
      if (space == ir::MemorySpace::Vec) {
        vec_tiles.emplace_back(var, tile_type);
      } else {
        cube_tiles.emplace_back(var, tile_type);
      }
    } else {
      shared_tiles.emplace_back(var, tile_type);
    }
  }

  // Helper lambda: emit auto& aliases for deduped tiles in a given memory space
  auto emit_deduped_aliases = [&](std::optional<ir::MemorySpace> filter_space) {
    for (const auto& [dup_var, kept_var] : deduped_aliases) {
      bool match = false;
      if (!filter_space.has_value()) {
        match = true;
      } else {
        for (const auto& [v, tile_type] : all_tiles) {
          if (v->name_ == dup_var->name_ && tile_type->memref_.has_value()) {
            auto space = (*tile_type->memref_)->memory_space_;
            if (space == filter_space.value()) match = true;
            break;
          }
        }
      }
      if (match) {
        std::string san_dup = context_.SanitizeName(dup_var);
        std::string san_kept = context_.SanitizeName(kept_var);
        emitter_.EmitLine("auto& " + san_dup + " = " + san_kept + ";");
        emitted_tile_aliases_.insert(san_dup);
      }
    }
  };

  // Emit shared tiles (outside any #if)
  if (!shared_tiles.empty()) {
    for (const auto& [var, tile_type] : shared_tiles) {
      GenerateTileTypeDeclaration(context_.SanitizeName(var), tile_type);
    }
    emitter_.EmitLine("");
  }

  // Emit Cube tiles inside #if defined(__DAV_CUBE__)
  if (!cube_tiles.empty()) {
    emitter_.EmitLine("#if defined(__DAV_CUBE__)");
    for (const auto& [var, tile_type] : cube_tiles) {
      GenerateTileTypeDeclaration(context_.SanitizeName(var), tile_type);
    }
    emit_deduped_aliases(ir::MemorySpace::Mat);
    emit_deduped_aliases(ir::MemorySpace::Scaling);
    emitter_.EmitLine("#endif");
    emitter_.EmitLine("");
  }

  // Emit Vec tiles inside #if defined(__DAV_VEC__)
  if (!vec_tiles.empty()) {
    emitter_.EmitLine("#if defined(__DAV_VEC__)");
    for (const auto& [var, tile_type] : vec_tiles) {
      GenerateTileTypeDeclaration(context_.SanitizeName(var), tile_type);
    }
    emit_deduped_aliases(ir::MemorySpace::Vec);
    emitter_.EmitLine("#endif");
    emitter_.EmitLine("");
  }
}

void CCECodegen::GenerateSinglePrologue(const ir::FunctionPtr& func, bool has_cross_sync) {
  EmitSingleFunctionSignature(func, has_cross_sync);
  auto section_shapes = CollectTensorAccessShapesPerSection(func->body_);
  EmitSingleGlobalTensors(func, section_shapes);
  EmitSingleTileDeclarations(func);
}

void CCECodegen::VisitStmt_(const ir::SectionStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null SectionStmt";

  // Flush any pending tile N-way before entering a new section
  FlushPendingTileNWay();

  if (single_file_mode_) {
    // Emit #if defined(__DAV_CUBE__) / #if defined(__DAV_VEC__)
    if (op->sectionKind_ == ir::SectionKind::Cube) {
      emitter_.EmitLine("#if defined(__DAV_CUBE__)");
    } else {
      emitter_.EmitLine("#if defined(__DAV_VEC__)");
    }
  }

  // Reset to pre-section state (Cube and Vector are compiled separately via #if guards).
  // Snapshot is saved on first section entry; restored on subsequent sections.
  if (!section_snapshot_saved_) {
    context_.SaveSnapshot();
    section_snapshot_saved_ = true;
  } else {
    context_.RestoreSnapshot();
    // On subsequent section entries, re-emit cross-section array declarations
    // (e.g. N-way event_id arrays) so they are visible inside this section's
    // #if guard. The Var bindings are already persistent via RegisterVarPersistent.
    for (const auto& decl : cross_section_decls_) {
      emitter_.EmitLine(decl);
    }
    if (!cross_section_decls_.empty()) {
      emitter_.EmitLine("");
    }
  }
  event_id_decls_.clear();
  // Keep event_id_decls_nway_ and event_id_names_used_ across sections so that
  // Vec section can reuse Cube-section N-way arrays (no duplicate declarations).
  tile_array_decls_.clear();
  tile_array_counter_ = 0;
  buffer_slot_decls_emitted_.clear();
  // Note: keep tile_addresses_ and emitted_tile_types_ — they contain prologue data
  // needed across all sections (tile TASSIGN addresses, type aliases).

  // Emit vector mask initialization at the start of Vec section (a3 only)
  if (op->sectionKind_ == ir::SectionKind::Vector && arch_ == "a3") {
    emitter_.EmitLine("set_mask_norm();");
    emitter_.EmitLine("set_vector_mask(-1, -1);");
    emitter_.EmitLine("");
  }

  // Visit the body
  if (op->body_) {
    auto prev_section = current_section_kind_;
    current_section_kind_ = op->sectionKind_;
    VisitStmt(op->body_);
    current_section_kind_ = prev_section;
  }

  if (single_file_mode_) {
    emitter_.EmitLine("#endif");
  }
}

bool CCECodegen::DetectCrossCoreSyncOps(const ir::StmtPtr& stmt) {
  if (!stmt) return false;

  class CrossCoreSyncDetector : public ir::IRVisitor {
      using ir::IRVisitor::VisitStmt_;
      using ir::IRVisitor::VisitExpr_;
   public:
    bool found = false;
    void VisitExpr_(const ir::CallPtr& op) override {
      const std::string& name = op->op_->name_;
      if (name == "system.set_cross_core" || name == "system.wait_cross_core" ||
          name == "system.set_cross_core_dyn" || name == "system.wait_cross_core_dyn") {
        found = true;
      }
      ir::IRVisitor::VisitExpr_(op);
    }
  };

  CrossCoreSyncDetector detector;
  detector.VisitStmt(stmt);
  return detector.found;
}

std::map<ir::VarPtr, ir::SectionKind> CCECodegen::CollectTileSections(const ir::StmtPtr& stmt) {
  if (!stmt) return {};

  class TileSectionCollector : public ir::IRVisitor {
      using ir::IRVisitor::VisitStmt_;
      using ir::IRVisitor::VisitExpr_;
   public:
    std::map<ir::VarPtr, ir::SectionKind> tile_sections;
    std::optional<ir::SectionKind> current_section;

    void VisitStmt_(const ir::SectionStmtPtr& op) override {
      auto prev = current_section;
      current_section = op->sectionKind_;
      ir::IRVisitor::VisitStmt_(op);
      current_section = prev;
    }

    void VisitStmt_(const ir::AssignStmtPtr& op) override {
      auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(op->var_->GetType());
      if (tile_type && current_section.has_value()) {
        tile_sections[op->var_] = *current_section;
      }
    }
  };

  TileSectionCollector collector;
  collector.VisitStmt(stmt);
  return collector.tile_sections;
}

// ========================================================================
// Phase 9 helper: Argument unpacking for GeneratePrologue
// ========================================================================

void CCECodegen::UnpackFunctionArguments(
    const ir::FunctionPtr& func,
    const std::map<std::string, std::vector<ir::ExprPtr>>& access_shapes) {
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    const std::string param_name = context_.SanitizeName(param);

    // tensor type parameter and type declaration
    if (auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType())) {
      // Extract element type
      std::string element_type = tensor_type->dtype_.ToCTypeString();

      // Emit argument unpacking via Tensor* indirection
      std::string tensor_var = param_name + "_tensor";
      emitter_.EmitLine("__gm__ Tensor* " + tensor_var + " = reinterpret_cast<__gm__ Tensor*>(args[" +
                        std::to_string(i) + "]);");
      emitter_.EmitLine("__gm__ " + element_type + "* " + param_name + " = reinterpret_cast<__gm__ " +
                        element_type + "*>(" + tensor_var + "->buffer.addr);");

      // Register parameter with "Global" suffix for use in operations
      const std::string global_name = param_name + "Global";
      context_.RegisterVar(param, global_name);

      // Look up access window shape for GlobalTensor Shape<>/Stride<> generation
      std::optional<std::vector<ir::ExprPtr>> access_shape;
      auto it = access_shapes.find(param->name_);
      if (it != access_shapes.end()) {
        access_shape = it->second;
      }

      GenerateGlobalTensorTypeDeclaration(global_name, tensor_type, param_name, tensor_var, access_shape);
    } else if (auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(param->GetType())) {
      // Generate scalar type declaration
      std::string cpp_type = scalar_type->dtype_.ToCTypeString();

      // Emit argument unpacking via union converter
      std::string conv_name = param_name + "_conv";
      emitter_.EmitLine("union { uint64_t u64; " + cpp_type + " val; } " + conv_name + ";");
      emitter_.EmitLine(conv_name + ".u64 = args[" + std::to_string(i) + "];");
      emitter_.EmitLine(cpp_type + " " + param_name + " = " + conv_name + ".val;");

      // Register scalar variable
      context_.RegisterVar(param, param_name);
    } else {
      throw ir::RuntimeError("Unsupported parameter type in function " + func->name_);
    }

    emitter_.EmitLine("");
  }
}

void CCECodegen::GeneratePrologue(const ir::FunctionPtr& func) {
  // Function signature
  emitter_.EmitLine(
      "extern \"C\" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args)");
  emitter_.EmitLine("{");
  emitter_.IncreaseIndent();

  emitter_.EmitLine("// Unpack arguments and type declarations");

  // Collect access window shapes so GlobalTensor Shape<> uses the block.load/store
  // window shape rather than the full tensor shape
  auto access_shapes = CollectTensorAccessShapes(func->body_);

  // Unpack arguments
  UnpackFunctionArguments(func, access_shapes);

  // Collect all TileType variables from function body
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> tile_vars;
  if (func->body_) {
    tile_vars = CollectTileVariables(func->body_);
  }

  // Generate Tile type definitions and allocations
  if (!tile_vars.empty()) {
    emitter_.EmitLine("// Tile type definitions and allocations");

    for (const auto& [var, tile_type] : tile_vars) {
      // Just use sanitized name (will be registered later in AssignStmt)
      const std::string var_name = context_.SanitizeName(var);

      // Generate Tile type and declaration (memref extracted automatically from tile_type)
      GenerateTileTypeDeclaration(var_name, tile_type);
    }

    emitter_.EmitLine("");
  }
}

void CCECodegen::GenerateBody(const ir::FunctionPtr& func) {
  if (func->body_) {
    VisitStmt(func->body_);
  }

  emitter_.DecreaseIndent();
  emitter_.EmitLine("}");
}

// ========================================================================
// Phase 8 helper: Tile-related assignment handling
// ========================================================================

bool CCECodegen::HandleTileRelatedAssignment(const ir::AssignStmtPtr& op) {
  // Skip TupleGetItemExpr assignments that extract Tiles/sub-tuples from make_tile tuples.
  // The tiles are already declared in the prologue with correct memref addresses.
  // Also skip assignments of TupleType/TileType values from make_tile no-ops.
  if (ir::As<ir::TupleGetItemExpr>(op->value_)) {
    auto result_type = op->value_->GetType();
    bool is_tile_related = ir::As<ir::TileType>(result_type) != nullptr;
    if (!is_tile_related) {
      // Check if TupleType ultimately contains TileTypes (nested make_tile result)
      if (auto tuple_type = ir::As<ir::TupleType>(result_type)) {
        is_tile_related = true;  // Any tuple from TupleGetItemExpr in tile context is make_tile related
      }
    }
    if (is_tile_related) {
      std::string var_name = context_.SanitizeName(op->var_);
      context_.RegisterVar(op->var_, var_name);
      // Propagate tile addresses from the prologue-declared tiles
      // e.g., q_mat_buf_0 �?_tuple_tmp_0_0 (which has address 0x0)
      // The TupleGetItemExpr naming gives us _tuple_tmp_0_0 etc.
      auto tge = ir::As<ir::TupleGetItemExpr>(op->value_);
      if (tge) {
        // Evaluate what the TupleGetItemExpr resolves to (e.g., "_tuple_tmp_0_0")
        VisitExpr(op->value_);
        std::string resolved = current_expr_value_;
        current_expr_value_ = "";
        if (tile_addresses_.count(resolved)) {
          tile_addresses_[var_name] = tile_addresses_[resolved];
          // If the resolved name differs from var_name and is a declared
          // prologue tile, emit a C++ reference alias so that user-created
          // aliases (e.g., `first_qk = qk_vec_buf[0]`) are valid C++
          // identifiers when used in TLOAD/TSTORE.
          if (resolved != var_name && emitted_tile_aliases_.find(var_name) == emitted_tile_aliases_.end()) {
            emitter_.EmitLine("auto& " + var_name + " = " + resolved + ";");
            emitted_tile_aliases_.insert(var_name);
          }
        }
      }
      current_target_var_ = "";
      return true;
    }
  }
  return false;
}

void CCECodegen::VisitStmt_(const ir::AssignStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null AssignStmt";
  INTERNAL_CHECK(op->var_ != nullptr) << "Internal error: AssignStmt has null variable";
  INTERNAL_CHECK(op->value_ != nullptr) << "Internal error: AssignStmt has null value";

  if (HandleTileRelatedAssignment(op)) return;

  // Also skip make_tile no-op assignments (value is a Call to make_tile �?returns "")
  // and TupleType assignments from SSA yield that carry tiles
  if (ir::As<ir::TupleType>(op->var_->GetType())) {
    std::string var_name = context_.SanitizeName(op->var_);
    context_.RegisterVar(op->var_, var_name);
    current_target_var_ = var_name;
    current_expr_value_ = "";
    VisitExpr(op->value_);
    // Don't emit assignment �?tuple is just a grouping of tiles
    current_expr_value_ = "";
    current_target_var_ = "";
    return;
  }

  // Sanitize and register the variable name
  std::string var_name = context_.SanitizeName(op->var_);
  context_.RegisterVar(op->var_, var_name);

  // Set context for expression visitor (dual-mode pattern)
  current_target_var_ = var_name;
  current_expr_value_ = "";

  // Dispatch to type-specific expression handler
  VisitExpr(op->value_);

  // If expression was non-Call (returned a value), emit assignment
  if (!current_expr_value_.empty()) {
    auto var_type = op->var_->GetType();
    // For tile variables assigned from another tile: register as alias (no code emitted).
    if (ir::As<ir::Var>(op->value_) &&
        (ir::As<ir::TileType>(var_type) || ir::As<ir::TupleType>(var_type))) {
      context_.RegisterVar(op->var_, current_expr_value_);
      if (tile_addresses_.count(current_expr_value_)) {
        tile_addresses_[var_name] = tile_addresses_[current_expr_value_];
      }
      current_expr_value_ = "";
      current_target_var_ = "";
      return;
    }
    // Copy propagation for scalar variable copies: `auto X = Y;` where Y is a simple variable
    // Alias X �?Y instead of emitting code. Resolve Y through existing aliases first.
    if (ir::As<ir::Var>(op->value_) && !ir::As<ir::TensorType>(var_type)) {
      std::string resolved = context_.ResolveAlias(current_expr_value_);
      context_.RegisterAlias(var_name, resolved);
      context_.RegisterVar(op->var_, resolved);
      current_expr_value_ = "";
      current_target_var_ = "";
      return;
    }

    // In VF scope, TileOffsetExpr results are __ubuf__ pointers
    if (in_vf_scope_ && ir::As<ir::TileOffsetExpr>(op->value_)) {
      emitter_.EmitLine("__ubuf__ uint8_t *" + var_name + " = " + current_expr_value_ + ";");
      vf_ptr_vars_.insert(var_name);
      current_expr_value_ = "";
    } else {
      // Scalar expression inlining: if a scalar SSA temp is read exactly once,
      // register it as an inline expression alias instead of emitting a declaration.
      // `auto task_id_11 = (task_id_6 + 1); task_id_6 = task_id_11;` becomes
      // `task_id_6 = (task_id_6 + 1);` with no intermediate variable.
      if (ir::As<ir::ScalarType>(var_type)) {
        auto it = var_read_counts_.find(op->var_->name_);
        if (it != var_read_counts_.end() && it->second <= 1) {
          context_.RegisterVar(op->var_, current_expr_value_);
          current_expr_value_ = "";
          current_target_var_ = "";
          return;
        }
      }
      emitter_.EmitLine("auto " + var_name + " = " + current_expr_value_ + ";");
      current_expr_value_ = "";
    }
  }

  // Clear context
  current_target_var_ = "";
}

void CCECodegen::VisitStmt_(const ir::EvalStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null EvalStmt";
  INTERNAL_CHECK(op->expr_ != nullptr) << "Internal error: EvalStmt has null expression";

  // EvalStmt: evaluate expression for side effects (e.g., sync operations)
  // Sync ops (set_flag, wait_flag, pipe_barrier) are registered with f_codegen_cce
  // and will be invoked via VisitExpr_(Call)
  VisitExpr(op->expr_);
}

void CCECodegen::VisitStmt_(const ir::ReturnStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ReturnStmt";
  // For void functions, we don't need to generate anything
  // The function will return implicitly at the closing brace
}

void CCECodegen::VisitStmt_(const ir::YieldStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null YieldStmt";

  if (op->value_.empty()) {
    return;  // No values to yield
  }

  // Visit each yielded expression and collect values
  std::vector<std::string> yielded_values;
  for (const auto& expr : op->value_) {
    VisitExpr(expr);
    yielded_values.push_back(current_expr_value_);
  }

  // Store in temporary buffer for ForStmt to pick up
  yield_buffer_ = yielded_values;
  current_expr_value_ = "";
}

std::vector<std::string> CCECodegen::ExtractYieldNames(const ir::StmtPtr& body) const {
  std::vector<std::string> yields;
  ir::YieldStmtPtr yield_stmt;
  if (auto y = ir::As<ir::YieldStmt>(body)) {
    yield_stmt = y;
  } else if (auto seq = ir::As<ir::SeqStmts>(body)) {
    if (!seq->stmts_.empty()) yield_stmt = ir::As<ir::YieldStmt>(seq->stmts_.back());
  }
  if (!yield_stmt) return yields;
  for (const auto& val : yield_stmt->value_) {
    if (auto var = ir::As<ir::Var>(val)) {
      yields.push_back(context_.SanitizeName(var));
    } else if (auto cint = ir::As<ir::ConstInt>(val)) {
      yields.push_back(std::to_string(cint->value_));
    } else if (auto iter_arg = ir::As<ir::IterArg>(val)) {
      yields.push_back(context_.SanitizeName(iter_arg));
    } else if (auto tge = ir::As<ir::TupleGetItemExpr>(val)) {
      if (auto tuple_var = ir::As<ir::Var>(tge->tuple_)) {
        yields.push_back(context_.SanitizeName(tuple_var) + "_" + std::to_string(tge->index_));
      } else if (auto tuple_iter = ir::As<ir::IterArg>(tge->tuple_)) {
        yields.push_back(context_.SanitizeName(tuple_iter) + "_" + std::to_string(tge->index_));
      } else if (auto make_tuple = ir::As<ir::MakeTuple>(tge->tuple_)) {
        size_t idx = static_cast<size_t>(tge->index_);
        if (idx < make_tuple->elements_.size()) {
          const auto& elem = make_tuple->elements_[idx];
          if (auto elem_var = ir::As<ir::Var>(elem)) {
            yields.push_back(context_.SanitizeName(elem_var));
          } else if (auto elem_cint = ir::As<ir::ConstInt>(elem)) {
            yields.push_back(std::to_string(elem_cint->value_));
          } else if (auto elem_iter = ir::As<ir::IterArg>(elem)) {
            yields.push_back(context_.SanitizeName(elem_iter));
          } else {
            return {};
          }
        } else {
          return {};
        }
      } else {
        return {};
      }
    } else {
      return {};
    }
  }
  return yields;
}

void CCECodegen::EmitYieldAssignments(const std::vector<ir::VarPtr>& return_vars,
                                      const std::vector<std::string>& target_names) {
  if (return_vars.empty() || yield_buffer_.empty()) return;
  for (size_t i = 0; i < return_vars.size(); ++i) {
    const auto& return_var = return_vars[i];
    std::string return_var_name = target_names[i];
    std::string yielded_value = yield_buffer_[i];

    auto return_type = return_var->GetType();
    std::string resolved_yield = context_.ResolveAlias(yielded_value);
    if ((ir::As<ir::TileType>(return_type) || ir::As<ir::TupleType>(return_type)) &&
        tile_addresses_.count(yielded_value)) {
      emitter_.EmitLine("TASSIGN(" + return_var_name + ", " + tile_addresses_[yielded_value] + ");");
      tile_addresses_[return_var_name] = tile_addresses_[yielded_value];
    } else if (return_var_name != resolved_yield) {
      emitter_.EmitLine(return_var_name + " = " + resolved_yield + ";");
    }

    if (std::dynamic_pointer_cast<const ir::TensorType>(return_type)) {
      std::string yielded_ptr = context_.GetPointer(resolved_yield);
      context_.RegisterPointer(return_var_name, yielded_ptr);
      if (!single_file_mode_) {
        std::string yielded_struct = context_.GetTensorStruct(resolved_yield);
        context_.RegisterTensorStruct(return_var_name, yielded_struct);
      }
    }
  }
  yield_buffer_.clear();
}

bool CCECodegen::TryEmitNWaySelect(const ir::IfStmtPtr& op) {
  if (op->returnVars_.empty() || !op->elseBody_.has_value()) return false;

  auto then_yields = ExtractYieldNames(op->thenBody_);
  // Walk the nested else chain: if(x==0){A} else{if(x==1){B} else{if(x==2){C}...}}
  std::vector<std::vector<std::string>> all_cases;
  bool nway_valid = !then_yields.empty() && then_yields.size() == op->returnVars_.size();
  if (nway_valid) {
    all_cases.push_back(then_yields);
    std::optional<ir::StmtPtr> cur_else = op->elseBody_;
    while (cur_else.has_value()) {
      ir::IfStmtPtr nested_if;
      if (auto direct_if = ir::As<ir::IfStmt>(*cur_else)) {
        nested_if = direct_if;
      } else if (auto seq = ir::As<ir::SeqStmts>(*cur_else)) {
        for (const auto& stmt : seq->stmts_) {
          if (auto nif = ir::As<ir::IfStmt>(stmt)) { nested_if = nif; break; }
        }
      }
      if (nested_if) {
        auto eq = ir::As<ir::Eq>(nested_if->condition_);
        auto rhs_c = eq ? ir::As<ir::ConstInt>(eq->right_) : nullptr;
        if (!rhs_c || rhs_c->value_ != static_cast<int64_t>(all_cases.size())) {
          nway_valid = false; break;
        }
        auto ys = ExtractYieldNames(nested_if->thenBody_);
        if (ys.size() != op->returnVars_.size()) { nway_valid = false; break; }
        all_cases.push_back(ys);
        cur_else = nested_if->elseBody_;
      } else {
        auto ys = ExtractYieldNames(*cur_else);
        if (ys.size() == op->returnVars_.size()) all_cases.push_back(ys);
        break;
      }
    }
  }

  if (!nway_valid || all_cases.size() < 2) return false;

  // Extract the LHS of condition (x == 0) to use as array index
  std::string index_expr;
  auto eq_op = ir::As<ir::Eq>(op->condition_);
  if (eq_op) {
    auto rhs_const = ir::As<ir::ConstInt>(eq_op->right_);
    if (rhs_const && rhs_const->value_ == 0) {
      VisitExpr(eq_op->left_);
      index_expr = current_expr_value_;
      current_expr_value_ = "";
    }
  }
  if (index_expr.empty()) return false;

  // Emit per-return-var arrays
  for (size_t i = 0; i < op->returnVars_.size(); ++i) {
    const auto& return_var = op->returnVars_[i];
    auto return_type = return_var->GetType();

    std::vector<std::string> vals;
    for (auto& c : all_cases) vals.push_back(c[i]);

    if (auto tile_type = ir::As<ir::TileType>(return_type)) {
      bool all_have_addr = true;
      for (auto& v : vals) {
        if (!tile_addresses_.count(v)) { all_have_addr = false; break; }
      }
      if (!all_have_addr) return false;

      // Dedup by element values + section; index_expr only affects which
      // element is read at the use site, not the array's contents.
      std::string tile_section_tag =
          (current_section_kind_ == ir::SectionKind::Cube) ? "cube:" : "vec:";
      std::string dedup_key = tile_section_tag;
      for (auto& v : vals) { dedup_key += v + ","; }
      std::string arr_name;
      if (tile_array_decls_.count(dedup_key)) {
        arr_name = tile_array_decls_[dedup_key];
        // If this tile array was already merged into a BufferSlot, use .tile accessor
        std::string access = arr_name + "[" + index_expr + "]";
        if (buffer_slot_decls_emitted_.count(dedup_key)) {
          access = arr_name + "[" + index_expr + "].tile";
        }
        context_.RegisterVarPersistent(return_var, access);
      } else {
        auto pos = vals[0].rfind('_');
        arr_name = (pos != std::string::npos) ? vals[0].substr(0, pos) : vals[0];
        arr_name += "_arr";
        if (tile_array_decls_.count(arr_name + "_dedup")) {
          arr_name += "_" + std::to_string(tile_array_counter_++);
        }
        tile_array_decls_[dedup_key] = arr_name;
        tile_array_decls_[arr_name + "_dedup"] = arr_name;
        std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);
        int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
        int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;
        std::string tile_type_str = type_converter_.ConvertTileType(tile_type, rows, cols);

        // Only defer NBuffer tiles (name starts with _nbuf_) for BufferSlot pairing.
        // Non-NBuffer tuples (e.g. global_sum_buf from make_tile Python tuples)
        // have no corresponding bid N-way and must emit immediately.
        bool is_nbuf_tile = (arr_name.find("_nbuf_") == 0);
        if (is_nbuf_tile) {
          FlushPendingTileNWay();
          pending_tile_nway_ = PendingTileNWay{
              index_expr, arr_name, tile_type_str, vals, return_var, dedup_key};
        } else {
          FlushPendingTileNWay();
          std::ostringstream arr_elems;
          for (size_t j = 0; j < vals.size(); ++j) {
            if (j > 0) arr_elems << ", ";
            arr_elems << vals[j];
          }
          std::string arr_decl = tile_type_str + " " + arr_name + "[] = {" + arr_elems.str() + "};";
          if (loop_depth_ > 0 || if_depth_ > 0) {
            loop_hoisted_decls_.push_back(arr_decl);
          } else {
            emitter_.EmitLine(arr_decl);
          }
        }
        context_.RegisterVarPersistent(return_var, arr_name + "[" + index_expr + "]");
      }

    } else if (ir::As<ir::ScalarType>(return_type)) {
      bool all_num = true;
      for (auto& v : vals) {
        if (v.empty() || (!std::isdigit(v[0]) && v[0] != '-')) { all_num = false; break; }
      }
      if (!all_num) return false;

      // Determine whether this return_var is used as a buf_id (Mutex) or an
      // event_id. buf_id uses ``uint8_t _bid_...`` to avoid semantic confusion
      // with event_id arrays (``event_t _eid_...``).
      bool is_buf_id = buf_id_var_names_.count(return_var->name_) > 0;

      // Try to merge with pending tile N-way into a BufferSlot struct array.
      if (is_buf_id && pending_tile_nway_.has_value() &&
          pending_tile_nway_->index_expr == index_expr &&
          pending_tile_nway_->tile_vals.size() == vals.size()) {
        auto& pending = *pending_tile_nway_;
        // Compute BufferSlot array name from tile array name
        std::string slot_name = pending.arr_name;
        auto suffix_pos = slot_name.find("_tiles_0_arr");
        if (suffix_pos != std::string::npos) {
          slot_name = slot_name.substr(0, suffix_pos);
        } else {
          // Fallback: strip _arr
          auto arr_pos = slot_name.rfind("_arr");
          if (arr_pos != std::string::npos) slot_name = slot_name.substr(0, arr_pos);
        }

        // Build BufferSlot dedup key
        std::string bs_section_tag =
            (current_section_kind_ == ir::SectionKind::Cube) ? "cube:" : "vec:";
        std::string bs_dedup_key = bs_section_tag + "bs:";
        for (auto& v : pending.tile_vals) bs_dedup_key += v + ",";
        bs_dedup_key += ":";
        for (auto& v : vals) bs_dedup_key += v + ",";

        if (!buffer_slot_decls_emitted_.count(bs_dedup_key)) {
          buffer_slot_decls_emitted_.insert(bs_dedup_key);
          // Also mark the tile dedup_key so dedup-hit path uses .tile accessor
          buffer_slot_decls_emitted_.insert(pending.dedup_key);
          // Update tile_array_decls_ to point to slot_name instead of arr_name
          tile_array_decls_[pending.dedup_key] = slot_name;

          // Emit BufferSlot template struct (once per file)
          if (!buffer_slot_struct_emitted_) {
            buffer_slot_struct_emitted_ = true;
            // The struct definition is emitted at class level (before function body)
            // but since we're inside function body, emit it here as a local struct.
            // C++ allows local struct definitions inside function scope.
          }

          // Build aggregate initializer: {{tile_0, bid_0}, {tile_1, bid_1}}
          std::ostringstream init;
          for (size_t j = 0; j < pending.tile_vals.size(); ++j) {
            if (j > 0) init << ", ";
            init << "{" << pending.tile_vals[j] << ", (uint8_t)" << vals[j] << "}";
          }
          std::string decl = "BufferSlot<" + pending.tile_type_str + "> " +
                             slot_name + "[] = {" + init.str() + "};";
          if (loop_depth_ > 0 || if_depth_ > 0) {
            loop_hoisted_decls_.push_back(decl);
          } else {
            emitter_.EmitLine(decl);
          }
        }

        // Re-register tile var to use .tile accessor
        context_.RegisterVarPersistent(pending.return_var,
                                       slot_name + "[" + index_expr + "].tile");
        // Register bid var to use .bid accessor
        context_.RegisterVar(return_var, slot_name + "[" + index_expr + "].bid");
        // Store bid dedup so subsequent hits reuse the BufferSlot
        std::string bid_section_tag =
            (current_section_kind_ == ir::SectionKind::Cube ? "cube:" : "vec:");
        std::string bid_dedup_key = bid_section_tag + "_bid:";
        for (auto& v : vals) bid_dedup_key += v + ",";
        event_id_decls_nway_[bid_dedup_key] = slot_name;
        // Mark this bid dedup key as a BufferSlot (uses .bid accessor)
        buffer_slot_decls_emitted_.insert("bid:" + bid_dedup_key);
        pending_tile_nway_ = std::nullopt;

      } else {
        // No pairing — flush pending tile as standalone array, emit bid/eid normally
        FlushPendingTileNWay();

        const std::string prefix = is_buf_id ? "_bid" : "_eid";
        const std::string elem_type = is_buf_id ? "uint8_t" : "event_t";
        const std::string cast_type = is_buf_id ? "(uint8_t)" : "(event_t)";

        // buf_id dedup keys are section-local (Cube/Vec have independent buf_id
        // spaces); event_id dedup keys are global (cross-core sync). Dedup on
        // element values only — index_expr affects the use site, not the array.
        std::string section_tag = is_buf_id ?
            (current_section_kind_ == ir::SectionKind::Cube ? "cube:" : "vec:") : "";
        std::string dedup_key = section_tag + prefix + ":";
        for (auto& v : vals) { dedup_key += v + ","; }

        std::string eid_name;
        if (event_id_decls_nway_.count(dedup_key)) {
          eid_name = event_id_decls_nway_[dedup_key];
          // If this dedup hit points to a BufferSlot, use .bid accessor
          if (buffer_slot_decls_emitted_.count("bid:" + dedup_key)) {
            context_.RegisterVar(return_var, eid_name + "[" + index_expr + "].bid");
            continue;
          }
        } else {
          eid_name = prefix;
          for (auto& v : vals) eid_name += "_" + v;
          std::string base_name = eid_name;
          while (event_id_names_used_.count(eid_name)) {
            eid_name = base_name + "_" + std::to_string(event_id_nway_counter_++);
          }
          event_id_names_used_.insert(eid_name);
          event_id_decls_nway_[dedup_key] = eid_name;
          std::ostringstream arr_elems;
          for (size_t j = 0; j < vals.size(); ++j) {
            if (j > 0) arr_elems << ", ";
            arr_elems << cast_type << vals[j];
          }
          std::string decl_line = "const " + elem_type + " " + eid_name +
                                  "[] = {" + arr_elems.str() + "};";
          if (!is_buf_id) {
            cross_section_decls_.push_back(decl_line);
          }
          if (loop_depth_ > 0 || if_depth_ > 0) {
            loop_hoisted_decls_.push_back(decl_line);
          } else {
            emitter_.EmitLine(decl_line);
          }
        }
        context_.RegisterVar(return_var, eid_name + "[" + index_expr + "]");
      }

    } else {
      return false;
    }
  }
  return true;
}

void CCECodegen::FlushPendingTileNWay() {
  if (!pending_tile_nway_.has_value()) return;
  auto& pending = *pending_tile_nway_;
  // Emit as standalone tile array (no BufferSlot pairing)
  std::ostringstream arr_elems;
  for (size_t j = 0; j < pending.tile_vals.size(); ++j) {
    if (j > 0) arr_elems << ", ";
    arr_elems << pending.tile_vals[j];
  }
  std::string arr_decl = pending.tile_type_str + " " + pending.arr_name +
                          "[] = {" + arr_elems.str() + "};";
  if (loop_depth_ > 0 || if_depth_ > 0) {
    loop_hoisted_decls_.push_back(arr_decl);
  } else {
    emitter_.EmitLine(arr_decl);
  }
  pending_tile_nway_ = std::nullopt;
}

bool CCECodegen::TryEmitIdentityElseIf(const ir::IfStmtPtr& op) {
  if (!op->elseBody_.has_value() || op->returnVars_.empty()) return false;

  // Extract else-yield Vars directly so we can resolve through name_to_cpp_
  // (not just alias_map_) — that catches cases where a Var was registered to
  // an expression like "arr[idx]" by the AssignVar->Tile path after N-way-select.
  auto extract_yield_vars = [](const ir::StmtPtr& body) -> std::vector<ir::VarPtr> {
    std::vector<ir::VarPtr> result;
    ir::YieldStmtPtr yield_stmt;
    if (auto y = ir::As<ir::YieldStmt>(body)) yield_stmt = y;
    else if (auto seq = ir::As<ir::SeqStmts>(body)) {
      if (!seq->stmts_.empty()) yield_stmt = ir::As<ir::YieldStmt>(seq->stmts_.back());
    }
    if (!yield_stmt) return {};
    for (const auto& val : yield_stmt->value_) {
      auto v = ir::As<ir::Var>(val);
      if (!v) return {};  // only handle pure-Var yields here
      result.push_back(v);
    }
    return result;
  };

  auto else_yield_vars = extract_yield_vars(*op->elseBody_);
  if (else_yield_vars.size() != op->returnVars_.size()) return false;

  // Check: the else body should be just a YieldStmt (no other side effects)
  auto else_body_ptr = *op->elseBody_;
  bool else_is_just_yield = false;
  if (ir::As<ir::YieldStmt>(else_body_ptr)) {
    else_is_just_yield = true;
  } else if (auto seq = ir::As<ir::SeqStmts>(else_body_ptr)) {
    else_is_just_yield = true;
    for (const auto& s : seq->stmts_) {
      if (!ir::As<ir::YieldStmt>(s)) { else_is_just_yield = false; break; }
    }
  }
  if (!else_is_just_yield) return false;

  // Save the pre-if resolved names by consulting name_to_cpp_ (GetVarName).
  std::vector<std::string> pre_if_names;
  for (const auto& v : else_yield_vars) {
    pre_if_names.push_back(context_.ResolveAlias(context_.GetVarName(v)));
  }

  // Reject if any pre_if_name is a complex expression (contains '[', '(', etc.):
  // identity-else treats pre_if_name as a C++ lvalue to assign into, but a
  // resolved name like `arr[idx]` (from N-way-select) would emit
  // `arr[idx] = new_val` which overwrites element storage instead of phi'ing.
  // Force full phi path for such cases so a real C++ lvalue is declared.
  auto is_simple_identifier = [](const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
      if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
    }
    return true;
  };
  for (const auto& name : pre_if_names) {
    if (!is_simple_identifier(name)) return false;
  }

  // Register return vars to the pre-if names
  for (size_t i = 0; i < op->returnVars_.size(); ++i) {
    context_.RegisterVar(op->returnVars_[i], pre_if_names[i]);
  }

  VisitExpr(op->condition_);
  std::string condition = current_expr_value_;
  current_expr_value_ = "";

  emitter_.EmitLine("if (" + condition + ") {");
  emitter_.IncreaseIndent();

  VisitStmt(op->thenBody_);
  EmitYieldAssignments(op->returnVars_, pre_if_names);

  emitter_.DecreaseIndent();
  emitter_.EmitLine("}");
  return true;
}

void CCECodegen::EmitFullPhiIf(const ir::IfStmtPtr& op) {
  // Declare and register return variables BEFORE the if statement
  for (const auto& return_var : op->returnVars_) {
    std::string return_var_name = context_.SanitizeName(return_var);
    context_.RegisterVar(return_var, return_var_name);

    if (auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(return_var->GetType())) {
      std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);
      int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
      int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;
      auto vs = ExtractValidShapeInfo(tile_type, rows, cols,
                                      [this](const ir::VarPtr& v) { return GetVarName(v); });
      std::string ctor_args = BuildTileCtorArgs(vs, rows, cols);
      std::string ctor_suffix = vs.needs_ctor ? ("(" + ctor_args + ")") : "";
      std::string type_alias_name = return_var_name + "Type";
      std::string tile_type_str = type_converter_.ConvertTileType(tile_type, rows, cols);
      if (loop_depth_ > 0) {
        loop_hoisted_decls_.push_back("using " + type_alias_name + " = " + tile_type_str + ";");
        loop_hoisted_decls_.push_back(type_alias_name + " " + return_var_name + ctor_suffix + ";");
      } else {
        emitter_.EmitLine("using " + type_alias_name + " = " + tile_type_str + ";");
        emitter_.EmitLine(type_alias_name + " " + return_var_name + ctor_suffix + ";");
      }
    } else if (auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(return_var->GetType())) {
      GenerateGlobalTensorTypeDeclaration(return_var_name, tensor_type);
    } else if (auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(return_var->GetType())) {
      std::string cpp_type = scalar_type->dtype_.ToCTypeString();
      emitter_.EmitLine(cpp_type + " " + return_var_name + ";");
    } else {
      throw ir::RuntimeError("Unsupported return_var type in IfStmt");
    }
  }

  VisitExpr(op->condition_);
  std::string condition = current_expr_value_;
  current_expr_value_ = "";

  emitter_.EmitLine("if (" + condition + ") {");
  emitter_.IncreaseIndent();
  VisitStmt(op->thenBody_);
  {
    std::vector<std::string> phi_names;
    for (const auto& rv : op->returnVars_) phi_names.push_back(context_.SanitizeName(rv));
    EmitYieldAssignments(op->returnVars_, phi_names);
  }
  emitter_.DecreaseIndent();

  if (op->elseBody_.has_value()) {
    emitter_.EmitLine("} else {");
    emitter_.IncreaseIndent();
    VisitStmt(*op->elseBody_);
    {
      std::vector<std::string> phi_names;
      for (const auto& rv : op->returnVars_) phi_names.push_back(context_.SanitizeName(rv));
      EmitYieldAssignments(op->returnVars_, phi_names);
    }
    emitter_.DecreaseIndent();
  }
  emitter_.EmitLine("}");
}

void CCECodegen::VisitStmt_(const ir::IfStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null IfStmt";
  INTERNAL_CHECK(op->condition_ != nullptr) << "Internal error: IfStmt has null condition";
  INTERNAL_CHECK(op->thenBody_ != nullptr) << "Internal error: IfStmt has null then_body";

  // Drop phi return_vars with no downstream consumer. The SSA pass
  // conservatively inserts phi nodes whenever a variable is re-assigned
  // across control flow, but if no one reads the phi output, emitting a
  // declaration + per-branch yield-assignment is pure dead code.
  ir::IfStmtPtr effective_op = op;
  if (!op->returnVars_.empty()) {
    bool any_used = false;
    for (const auto& rv : op->returnVars_) {
      if (rv && var_read_names_.count(rv->name_)) { any_used = true; break; }
    }
    if (!any_used) {
      // An else-body consisting only of YieldStmts also becomes dead once
      // return_vars are dropped — the yields have no consumer.
      auto else_is_only_yields = [](const ir::StmtPtr& s) -> bool {
        if (!s) return false;
        if (ir::As<ir::YieldStmt>(s)) return true;
        if (auto seq = ir::As<ir::SeqStmts>(s)) {
          for (const auto& st : seq->stmts_) {
            if (!ir::As<ir::YieldStmt>(st)) return false;
          }
          return !seq->stmts_.empty();
        }
        return false;
      };
      std::optional<ir::StmtPtr> new_else = op->elseBody_;
      if (new_else.has_value() && else_is_only_yields(*new_else)) {
        new_else = std::nullopt;
      }
      effective_op = std::make_shared<ir::IfStmt>(
          op->condition_, op->thenBody_, new_else,
          std::vector<ir::VarPtr>{}, op->span_);
    }
  }

  // N-way select optimization: if(x==0){A} else{if(x==1){B}...} → array[x]
  if (TryEmitNWaySelect(effective_op)) return;

  // N-way didn't match — flush any pending tile that was waiting for a bid pair
  FlushPendingTileNWay();

  // If-level hoisting: buffer output so array decls can be hoisted before the if
  bool is_outermost_if = (loop_depth_ == 0 && if_depth_ == 0);
  if_depth_++;

  std::string pre_if_code;
  int saved_if_indent = emitter_.GetIndentLevel();
  if (is_outermost_if) {
    pre_if_code = emitter_.GetCode();
    emitter_.Clear();
    emitter_.SetIndentLevel(saved_if_indent);
  }

  // Try identity-else optimization, fall back to full phi codegen
  if (!TryEmitIdentityElseIf(effective_op)) {
    EmitFullPhiIf(effective_op);
  }

  if_depth_--;

  // Insert hoisted declarations before the if statement
  if (is_outermost_if) {
    std::string if_code = emitter_.GetCode();
    emitter_.Clear();
    emitter_.SetIndentLevel(saved_if_indent);
    emitter_.EmitRaw(pre_if_code);

    if (!loop_hoisted_decls_.empty()) {
      for (const auto& decl : loop_hoisted_decls_) {
        emitter_.EmitLine(decl);
      }
      emitter_.EmitLine("");
      loop_hoisted_decls_.clear();
    }

    emitter_.EmitRaw(if_code);
  }
}

// ========================================================================
// Phase 5 helpers: ForStmt iter-arg registration and yield assignments
// ========================================================================

std::vector<std::string> CCECodegen::RegisterForIterArgs(const ir::ForStmtPtr& op) {
  std::vector<std::string> iter_arg_names;
  if (op->iterArgs_.empty()) return iter_arg_names;

  bool any_emitted = false;
  for (auto& iter_arg : op->iterArgs_) {
    std::string iter_arg_name = context_.SanitizeName(iter_arg);

    // Evaluate init value
    VisitExpr(iter_arg->initValue_);
    std::string init_value = current_expr_value_;
    current_expr_value_ = "";

    // If initializing from a tensor variable, inherit both pointer and Tensor struct mappings
    auto init_var = std::dynamic_pointer_cast<const ir::Var>(iter_arg->initValue_);
    if (init_var && std::dynamic_pointer_cast<const ir::TensorType>(init_var->GetType())) {
      std::string init_var_name = context_.GetVarName(init_var);
      std::string init_ptr = context_.GetPointer(init_var_name);
      context_.RegisterPointer(iter_arg_name, init_ptr);

      if (!single_file_mode_) {
        std::string init_struct = context_.GetTensorStruct(init_var_name);
        context_.RegisterTensorStruct(iter_arg_name, init_struct);
      }
    }

    // For loop-carried state, only reuse an existing slot when the init
    // resolves to a writable lvalue. Literals / expressions like `0` or
    // `(x + 1)` must materialize a real loop-local variable.
    std::string resolved_init = context_.ResolveAlias(init_value);
    bool is_simple_var_copy = (init_var != nullptr) &&
                              !std::dynamic_pointer_cast<const ir::TensorType>(init_var->GetType()) &&
                              !std::dynamic_pointer_cast<const ir::TileType>(init_var->GetType()) &&
                              !std::dynamic_pointer_cast<const ir::TupleType>(init_var->GetType());
    bool can_reuse_writable_slot = is_simple_var_copy && IsWritableLValueExpr(resolved_init);
    // In single-file mode, skip copy propagation if init is a cross-section variable
    // (auto-registered after snapshot restore) to avoid aliasing to undeclared names.
    if (can_reuse_writable_slot && single_file_mode_ && context_.IsAutoRegistered(resolved_init)) {
      can_reuse_writable_slot = false;
    }

    if (can_reuse_writable_slot) {
      // Alias: iter_arg_name �?resolved init var.  No code emitted.
      context_.RegisterAlias(iter_arg_name, resolved_init);
      context_.RegisterVar(iter_arg, resolved_init);  // make IR var resolve to canonical name
      iter_arg_names.push_back(resolved_init);
    } else {
      // Real declaration needed
      context_.RegisterVar(iter_arg, iter_arg_name);
      iter_arg_names.push_back(iter_arg_name);
      if (!any_emitted) {
        any_emitted = true;
      }
      // If init references a cross-section variable, substitute with 0
      std::string safe_init = init_value;
      if (single_file_mode_ && context_.IsAutoRegistered(init_value)) {
        safe_init = "0";
      }
      emitter_.EmitLine("auto " + iter_arg_name + " = " + safe_init + ";");
    }
  }
  if (any_emitted) {
    emitter_.EmitLine("");
  }
  return iter_arg_names;
}

void CCECodegen::EmitForYieldAssignments(const std::vector<std::string>& iter_arg_names) {
  if (yield_buffer_.empty()) return;

  CHECK(yield_buffer_.size() == iter_arg_names.size())
      << "Yielded " << yield_buffer_.size() << " values but expected " << iter_arg_names.size();

  for (size_t i = 0; i < iter_arg_names.size(); ++i) {
    std::string lhs = iter_arg_names[i];
    std::string rhs = context_.ResolveAlias(yield_buffer_[i]);
    if (lhs == rhs) {
      continue;  // Self-assignment after alias resolution �?skip
    }
    // For tiles: use TASSIGN instead of operator= to transfer hardware address
    std::string tile_source = tile_addresses_.count(yield_buffer_[i]) ? yield_buffer_[i] : rhs;
    if (tile_addresses_.count(tile_source)) {
      emitter_.EmitLine("TASSIGN(" + lhs + ", " + tile_addresses_[tile_source] + ");");
      tile_addresses_[lhs] = tile_addresses_[tile_source];
    } else {
      emitter_.EmitLine(lhs + " = " + rhs + ";");
    }
  }
  yield_buffer_.clear();
}

void CCECodegen::VisitStmt_(const ir::ForStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ForStmt";
  INTERNAL_CHECK(op->loopVar_ != nullptr) << "Internal error: ForStmt has null loop_var";
  INTERNAL_CHECK(op->start_ != nullptr) << "Internal error: ForStmt has null start";
  INTERNAL_CHECK(op->stop_ != nullptr) << "Internal error: ForStmt has null stop";
  INTERNAL_CHECK(op->step_ != nullptr) << "Internal error: ForStmt has null step";
  INTERNAL_CHECK(op->body_ != nullptr) << "Internal error: ForStmt has null body";

  // Check consistency: iter_args and return_vars must have same size
  CHECK(op->iterArgs_.size() == op->returnVars_.size())
      << "ForStmt iter_args size (" << op->iterArgs_.size() << ") must equal return_vars size ("
      << op->returnVars_.size() << ")";

  if (op->kind_ == ir::ForKind::Unroll) {
    LOG_WARN << "ForKind::Unroll loop was not expanded before codegen; "
                "generating sequential loop as fallback";
  }

  // --- Early single-iteration detection ---
  auto start_ci = ir::As<ir::ConstInt>(op->start_);
  auto stop_ci = ir::As<ir::ConstInt>(op->stop_);
  auto step_ci = ir::As<ir::ConstInt>(op->step_);
  bool is_single_iter = (start_ci && stop_ci && step_ci &&
                         start_ci->value_ == 0 && stop_ci->value_ == 1 && step_ci->value_ == 1);

  // Register loop variable
  std::string loop_var_name;
  if (is_single_iter) {
    // Single-iteration: register loop var as constant "0"
    loop_var_name = "0";
    context_.RegisterVar(op->loopVar_, "0");
  } else {
    loop_var_name = context_.SanitizeName(op->loopVar_);
    context_.RegisterVar(op->loopVar_, loop_var_name);
  }

  // Register iteration arguments (loop-carried values)
  std::vector<std::string> iter_arg_names = RegisterForIterArgs(op);

  // Evaluate loop range
  VisitExpr(op->start_);
  std::string start = current_expr_value_;
  current_expr_value_ = "";

  VisitExpr(op->stop_);
  std::string stop = current_expr_value_;
  current_expr_value_ = "";

  VisitExpr(op->step_);
  std::string step = current_expr_value_;
  current_expr_value_ = "";

  // --- Single-iteration loop unrolling ---
  if (is_single_iter) {
    // Visit loop body directly (no loop wrapper)
    yield_buffer_.clear();
    VisitStmt(op->body_);
    yield_buffer_.clear();

    // Register return variables with same names as iter_args
    if (!op->returnVars_.empty()) {
      for (size_t i = 0; i < op->returnVars_.size(); ++i) {
        const auto& return_var = op->returnVars_[i];
        if (i < iter_arg_names.size()) {
          context_.RegisterVar(return_var, iter_arg_names[i]);
        } else {
          throw ir::RuntimeError("ForStmt return_var has no corresponding iter_arg");
        }
      }
    }
    return;
  }

  // --- Emit for-loop with hoisting ---
  EmitForLoopWithHoisting(op, loop_var_name, iter_arg_names, start, stop, step);
}

void CCECodegen::EmitForLoopWithHoisting(
    const ir::ForStmtPtr& op,
    const std::string& loop_var_name,
    const std::vector<std::string>& iter_arg_names,
    const std::string& start, const std::string& stop, const std::string& step) {
  bool is_outermost_loop = (loop_depth_ == 0);
  loop_depth_++;
  size_t hoist_start_idx = loop_hoisted_decls_.size();

  std::string pre_for_code;
  int saved_indent = emitter_.GetIndentLevel();
  if (is_outermost_loop) {
    pre_for_code = emitter_.GetCode();
    emitter_.Clear();
    emitter_.SetIndentLevel(saved_indent);
  }

  // In __VEC_SCOPE__, bisheng requires uint16_t loop variables
  std::string loop_type = in_vf_scope_ ? "uint16_t" : "uint64_t";
  emitter_.EmitLine("for (" + loop_type + " " + loop_var_name + " = " + start + "; " + loop_var_name + " < " + stop +
                    "; " + loop_var_name + " += " + step + ") {");
  emitter_.IncreaseIndent();

  yield_buffer_.clear();
  VisitStmt(op->body_);

  if (!op->iterArgs_.empty()) {
    EmitForYieldAssignments(iter_arg_names);
  }

  emitter_.DecreaseIndent();
  emitter_.EmitLine("}");

  loop_depth_--;

  // Insert hoisted declarations before the for-loop
  if (is_outermost_loop) {
    std::string for_code = emitter_.GetCode();
    emitter_.Clear();
    emitter_.SetIndentLevel(saved_indent);
    emitter_.EmitRaw(pre_for_code);

    if (loop_hoisted_decls_.size() > hoist_start_idx) {
      for (size_t i = hoist_start_idx; i < loop_hoisted_decls_.size(); ++i) {
        emitter_.EmitLine(loop_hoisted_decls_[i]);
      }
      emitter_.EmitLine("");
      loop_hoisted_decls_.resize(hoist_start_idx);
    }

    emitter_.EmitRaw(for_code);
  }

  // Register return variables with same names as iter_args
  if (!op->returnVars_.empty()) {
    for (size_t i = 0; i < op->returnVars_.size(); ++i) {
      const auto& return_var = op->returnVars_[i];
      if (i < iter_arg_names.size()) {
        context_.RegisterVar(return_var, iter_arg_names[i]);
      } else {
        throw ir::RuntimeError("ForStmt return_var has no corresponding iter_arg");
      }
    }
  }
}

void CCECodegen::VisitStmt_(const ir::WhileStmtPtr& op) {
  throw ir::RuntimeError("WhileStmt codegen not yet implemented");
}

// ========================================================================
// Expression Visitor Methods - Dual-Mode Pattern
// ========================================================================
// - Statement-Emitting Mode (Call): Uses current_target_var_, emits instructions
// - Value-Returning Mode (others): Sets current_expr_value_ with inline C++ code

// ---- Leaf Nodes ----

void CCECodegen::VisitExpr_(const ir::VarPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Var";
  current_expr_value_ = context_.GetVarName(op);
}

void CCECodegen::VisitExpr_(const ir::IterArgPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null IterArg";
  // IterArg inherits from Var, treated same way
  current_expr_value_ = context_.GetVarName(std::dynamic_pointer_cast<const ir::Var>(op));
}

void CCECodegen::VisitExpr_(const ir::ConstIntPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstInt";
  current_expr_value_ = std::to_string(op->value_);
}

void CCECodegen::VisitExpr_(const ir::ConstFloatPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstFloat";
  current_expr_value_ = std::to_string(op->value_);
}

void CCECodegen::VisitExpr_(const ir::ConstBoolPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstBool";
  current_expr_value_ = op->value_ ? "true" : "false";
}

void CCECodegen::VisitExpr_(const ir::TupleGetItemExprPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null TupleGetItemExpr";

  // If the tuple is a MakeTuple literal, just evaluate the specific element directly.
  // This handles constant tuples like event_ids = (0, 1).
  if (auto make_tuple = ir::As<ir::MakeTuple>(op->tuple_)) {
    CHECK(op->index_ >= 0 && op->index_ < static_cast<int>(make_tuple->elements_.size()))
        << "TupleGetItemExpr index " << op->index_ << " out of bounds";
    VisitExpr(make_tuple->elements_[op->index_]);
    return;
  }

  VisitExpr(op->tuple_);
  std::string tuple_name = current_expr_value_;

  auto tuple_type = op->tuple_->GetType();
  // For TupleType/TileType tuples (from make_tile, double-buffer patterns):
  // Elements are declared as separate tiles with "_idx" suffix in prologue.
  if (std::dynamic_pointer_cast<const ir::TupleType>(tuple_type) ||
      std::dynamic_pointer_cast<const ir::TileType>(tuple_type)) {
    current_expr_value_ = tuple_name + "_" + std::to_string(op->index_);
    return;
  }

  current_expr_value_ = tuple_name + "[" + std::to_string(op->index_) + "]";
}

void CCECodegen::VisitExpr_(const ir::TileOffsetExprPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null TileOffsetExpr";

  // Get base tile name and offset expression
  std::string base_tile = GetExprAsCode(op->tile_);
  std::string offset_expr = GetExprAsCode(op->offset_);

  // Compute element size in bytes from dtype
  auto tile_type = ir::As<ir::TileType>(op->tile_->GetType());
  INTERNAL_CHECK(tile_type != nullptr) << "TileOffsetExpr tile must have TileType";
  int elem_bytes = static_cast<int>(tile_type->dtype_.GetBit() / 8);
  if (elem_bytes == 0) elem_bytes = 1;  // guard sub-byte types

  // In VF scope: generate __ubuf__ pointer expression (inline, no temp variable)
  if (in_vf_scope_) {
    current_expr_value_ = "((__ubuf__ uint8_t *)" + base_tile + ".data() + (" +
                          offset_expr + ") * " + std::to_string(elem_bytes) + ")";
    return;
  }

  // Get base tile address — try tile_addresses_ first, then extract from TileType memref
  std::string base_addr;
  auto addr_it = tile_addresses_.find(base_tile);
  if (addr_it != tile_addresses_.end()) {
    base_addr = addr_it->second;
  } else {
    // Fallback: extract address from TileType's memref
    INTERNAL_CHECK(tile_type != nullptr && tile_type->memref_.has_value())
        << "TileOffsetExpr: base tile '" << base_tile << "' has no address info";
    int64_t addr = ExtractConstInt((*tile_type->memref_)->addr_);
    base_addr = FormatAddressHex(addr);
  }

  // Generate unique temp tile name
  std::string temp_name = base_tile + "_eoff_" + std::to_string(tile_offset_counter_++);

  // Emit: declare temp tile of same type, then TASSIGN with offset address
  // Use direct type + TASSIGN instead of `auto temp = base;` to avoid undeclared base tile issues.
  std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);
  int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
  int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;
  auto vs = ExtractValidShapeInfo(tile_type, rows, cols,
                                  [this](const ir::VarPtr& v) { return GetVarName(v); });
  std::string ctor_args = BuildTileCtorArgs(vs, rows, cols);
  std::string ctor_suffix = vs.needs_ctor ? ("(" + ctor_args + ")") : "";
  std::string type_str = type_converter_.ConvertTileType(tile_type, rows, cols);
  std::string temp_addr = base_addr + " + (" + offset_expr + ") * " + std::to_string(elem_bytes);
  emitter_.EmitLine(type_str + " " + temp_name + ctor_suffix + "; " +
                    "TASSIGN(" + temp_name + ", " + temp_addr + ");");
  tile_addresses_[temp_name] = temp_addr;

  current_expr_value_ = temp_name;
}

// ========================================================================
// CodegenBase interface and CCE-specific helper methods
// ========================================================================

std::string CCECodegen::GetExprAsCode(const ir::ExprPtr& expr) {
  VisitExpr(expr);
  return current_expr_value_;
}

void CCECodegen::Emit(const std::string& line) { emitter_.EmitLine(line); }

std::string CCECodegen::GetTypeString(const ir::DataType& dtype) const { return dtype.ToCTypeString(); }

int64_t CCECodegen::GetConstIntValue(const ir::ExprPtr& expr) { return ExtractConstInt(expr); }

std::string CCECodegen::GetVarName(const ir::VarPtr& var) { return context_.GetVarName(var); }

std::string CCECodegen::GetPointer(const std::string& var_name) { return context_.GetPointer(var_name); }

std::string CCECodegen::ComputeIRBasedOffset(const ir::TensorTypePtr& tensor_type,
                                             const ir::MakeTuplePtr& offsets) {
  // Compute row-major strides from IR tensor shape and build offset expression.
  // For tensor shape [d0, d1, ..., dn-1]:
  //   stride[i] = d_{i+1} * d_{i+2} * ... * d_{n-1}
  //   stride[n-1] = 1
  //   offset = off[0]*stride[0] + off[1]*stride[1] + ...
  size_t ndim = tensor_type->shape_.size();
  CHECK(offsets->elements_.size() == ndim)
      << "Offset dimensions (" << offsets->elements_.size() << ") != tensor dimensions (" << ndim << ")";

  std::ostringstream result;
  result << "(";
  bool first = true;
  for (size_t i = 0; i < ndim; ++i) {
    std::string off_expr = GetExprAsCode(offsets->elements_[i]);

    // Build stride as product of shape[i+1..n-1]
    // For the last dimension, stride = 1, so just add the offset directly
    if (!first) result << " + ";
    first = false;
    result << off_expr;

    for (size_t j = i + 1; j < ndim; ++j) {
      result << " * " << GetExprAsCode(tensor_type->shape_[j]);
    }
  }
  result << ")";
  return result.str();
}

void CCECodegen::RegisterOutputPointer(const std::string& output_var_name,
                                       const std::string& tensor_var_name) {
  context_.RegisterPointer(output_var_name, tensor_var_name);
}

std::string CCECodegen::GetTensorStruct(const std::string& var_name) {
  return context_.GetTensorStruct(var_name);
}

void CCECodegen::RegisterOutputTensorStruct(const std::string& output_var_name,
                                            const std::string& tensor_var_name) {
  context_.RegisterTensorStruct(output_var_name, tensor_var_name);
}

// ========================================================================
// Call Expression Visitor (uses operator registry codegen functions)
// ========================================================================

void CCECodegen::VisitExpr_(const ir::CallPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Call";

  CHECK(backend_ != nullptr) << "CCE backend must not be null";
  const auto* op_info = backend_->GetOpInfo(op->op_->name_);
  if (op_info == nullptr) {
    ThrowNoCodegenForCall(op->op_->name_);
  }
  std::string result = op_info->codegen_func(op, *this);
  current_expr_value_ = result;
}

// ---- Binary Operators ----

#define IMPLEMENT_BINARY_OP(OpType, OpName, CppOp)                        \
  void CCECodegen::VisitExpr_(const ir::OpType##Ptr& op) {                \
    INTERNAL_CHECK(op != nullptr) << "Internal error: null " << (OpName); \
    VisitExpr(op->left_);                                                 \
    std::string left = current_expr_value_;                               \
    VisitExpr(op->right_);                                                \
    std::string right = current_expr_value_;                              \
    current_expr_value_ = "(" + left + " " + (CppOp) + " " + right + ")"; \
  }

// Arithmetic operators
IMPLEMENT_BINARY_OP(Add, "Add", "+")
IMPLEMENT_BINARY_OP(Sub, "Sub", "-")
IMPLEMENT_BINARY_OP(Mul, "Mul", "*")
IMPLEMENT_BINARY_OP(FloorDiv, "FloorDiv", "/")
IMPLEMENT_BINARY_OP(FloorMod, "FloorMod", "%")
IMPLEMENT_BINARY_OP(FloatDiv, "FloatDiv", "/")

// Comparison operators
IMPLEMENT_BINARY_OP(Eq, "Eq", "==")
IMPLEMENT_BINARY_OP(Ne, "Ne", "!=")
IMPLEMENT_BINARY_OP(Lt, "Lt", "<")
IMPLEMENT_BINARY_OP(Le, "Le", "<=")
IMPLEMENT_BINARY_OP(Gt, "Gt", ">")
IMPLEMENT_BINARY_OP(Ge, "Ge", ">=")

// Logical operators
IMPLEMENT_BINARY_OP(And, "And", "&&")
IMPLEMENT_BINARY_OP(Or, "Or", "||")
IMPLEMENT_BINARY_OP(Xor, "Xor", "^")

// Bitwise operators
IMPLEMENT_BINARY_OP(BitAnd, "BitAnd", "&")
IMPLEMENT_BINARY_OP(BitOr, "BitOr", "|")
IMPLEMENT_BINARY_OP(BitXor, "BitXor", "^")
IMPLEMENT_BINARY_OP(BitShiftLeft, "BitShiftLeft", "<<")
IMPLEMENT_BINARY_OP(BitShiftRight, "BitShiftRight", ">>")

#undef IMPLEMENT_BINARY_OP

// Special binary operators (function calls)
void CCECodegen::VisitExpr_(const ir::MinPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Min";
  VisitExpr(op->left_);
  std::string left = current_expr_value_;
  VisitExpr(op->right_);
  std::string right = current_expr_value_;
  current_expr_value_ = "min(" + left + ", " + right + ")";
}

void CCECodegen::VisitExpr_(const ir::MaxPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Max";
  VisitExpr(op->left_);
  std::string left = current_expr_value_;
  VisitExpr(op->right_);
  std::string right = current_expr_value_;
  current_expr_value_ = "max(" + left + ", " + right + ")";
}

void CCECodegen::VisitExpr_(const ir::PowPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Pow";
  VisitExpr(op->left_);
  std::string left = current_expr_value_;
  VisitExpr(op->right_);
  std::string right = current_expr_value_;
  current_expr_value_ = "pow(" + left + ", " + right + ")";
}

// ---- Unary Operators ----

#define IMPLEMENT_UNARY_OP(OpType, OpName, CppOp)                                 \
  void CCECodegen::VisitExpr_(const ir::OpType##Ptr& op) {                        \
    INTERNAL_CHECK(op != nullptr) << "Internal error: null " << (OpName);         \
    VisitExpr(op->operand_);                                                      \
    current_expr_value_ = std::string("(") + (CppOp) + current_expr_value_ + ")"; \
  }

IMPLEMENT_UNARY_OP(Neg, "Neg", "-")
IMPLEMENT_UNARY_OP(Not, "Not", "!")
IMPLEMENT_UNARY_OP(BitNot, "BitNot", "~")

#undef IMPLEMENT_UNARY_OP

// Special unary operators
void CCECodegen::VisitExpr_(const ir::AbsPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Abs";
  VisitExpr(op->operand_);
  std::string operand = current_expr_value_;
  current_expr_value_ = "abs(" + operand + ")";
}

void CCECodegen::VisitExpr_(const ir::CastPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Cast";
  VisitExpr(op->operand_);
  std::string operand = current_expr_value_;

  auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(op->GetType());
  CHECK(scalar_type != nullptr) << "Cast target must be ScalarType";

  std::string cpp_type = scalar_type->dtype_.ToCTypeString();
  current_expr_value_ = "((" + cpp_type + ")" + operand + ")";
}

// ========================================================================
// End of Expression Visitor Methods
// ========================================================================

int64_t CCECodegen::ExtractConstInt(const ir::ExprPtr& expr) {
  auto const_int = std::dynamic_pointer_cast<const ir::ConstInt>(expr);
  CHECK(const_int != nullptr) << "Expected constant integer expression";
  return const_int->value_;
}

namespace {

/**
 * @brief Helper visitor for collecting TileType variables from IR
 *
 * Traverses the IR tree and collects all variables with TileType.
 * Uses the visitor pattern for clean, extensible traversal.
 */
class TileCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitStmt_;
    using ir::IRVisitor::VisitExpr_;
 public:
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> tile_vars_;

  // Extract the first TileType from a possibly nested TupleType
  static ir::TileTypePtr ExtractLeafTileType(const ir::TypePtr& type) {
    if (auto tile = std::dynamic_pointer_cast<const ir::TileType>(type)) {
      return tile;
    }
    if (auto tuple = std::dynamic_pointer_cast<const ir::TupleType>(type)) {
      for (const auto& elem : tuple->types_) {
        auto result = ExtractLeafTileType(elem);
        if (result) return result;
      }
    }
    return nullptr;
  }

  // Recursively collect all TileTypes from a type (handles TupleType nesting)
  static void CollectTileTypesFromType(const ir::TypePtr& type, std::vector<ir::TileTypePtr>& result) {
    if (auto tile = std::dynamic_pointer_cast<const ir::TileType>(type)) {
      result.push_back(tile);
    } else if (auto tuple = std::dynamic_pointer_cast<const ir::TupleType>(type)) {
      for (const auto& elem : tuple->types_) {
        CollectTileTypesFromType(elem, result);
      }
    }
  }

  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    auto var_type = op->var_->GetType();
    // Direct TileType variable
    if (auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(var_type)) {
      tile_vars_.emplace_back(op->var_, tile_type);
      return;
    }
    // TupleType variable (from make_tile or double-buffer patterns):
    // Expand each TileType element as a separate tile with indexed name.
    if (std::dynamic_pointer_cast<const ir::TupleType>(var_type)) {
      std::vector<ir::TileTypePtr> tile_types;
      CollectTileTypesFromType(var_type, tile_types);
      for (size_t i = 0; i < tile_types.size(); ++i) {
        // Create a synthetic Var for each tuple element: varname_0, varname_1, ...
        auto elem_var = std::make_shared<ir::Var>(
            op->var_->name_ + "_" + std::to_string(i), tile_types[i], op->var_->span_);
        tile_vars_.emplace_back(elem_var, tile_types[i]);
      }
    }
  }
};

/**
 * @brief Helper visitor for collecting tensor access shapes from block.load/store
 *
 * Traverses the IR tree to find block.load/block.store calls
 * and extracts the access window shapes (shapes_tuple) for each tensor parameter.
 * The GlobalTensor shape should match the access window, not the full tensor shape.
 */
class TensorAccessShapeCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitStmt_;
    using ir::IRVisitor::VisitExpr_;
 public:
  // Per-section access shapes: cube_access_shapes_ and vec_access_shapes_
  // store the first access shape found within each section for each tensor.
  // access_shapes_ stores shapes for tensors accessed outside any section.
  std::map<std::string, std::vector<ir::ExprPtr>> access_shapes_;
  std::map<std::string, std::vector<ir::ExprPtr>> cube_access_shapes_;
  std::map<std::string, std::vector<ir::ExprPtr>> vec_access_shapes_;
  std::map<std::string, std::vector<int>> access_tile_dims_;
  std::map<std::string, std::vector<int>> cube_access_tile_dims_;
  std::map<std::string, std::vector<int>> vec_access_tile_dims_;
  std::set<std::string> dn_tensors_;  // tensor names loaded with layout="dn"

  void VisitStmt_(const ir::SectionStmtPtr& op) override {
    auto prev = current_section_;
    current_section_ = op->sectionKind_;
    ir::IRVisitor::VisitStmt_(op);
    current_section_ = prev;
  }

  void VisitExpr_(const ir::CallPtr& op) override {
    const std::string& op_name = op->op_->name_;

    // Determine tensor arg index and tile arg index:
    // block.load: tensor at arg[0], shapes at arg[2], tile at arg[3]
    // block.store: tensor at arg[3], shapes at arg[2], tile at arg[0]
    // manual.load: tensor at arg[0], tile at arg[2] (no shapes arg)
    // manual.store: tensor at arg[2], tile at arg[0] (no shapes arg)
    // manual.store_fp: tensor at arg[3], fp_tile at arg[1], tile at arg[0]
    int tensor_arg_idx = -1;
    int shapes_arg_idx = -1;
    int tile_arg_idx = -1;
    if (op_name == "block.load") {
      tensor_arg_idx = 0; shapes_arg_idx = 2; tile_arg_idx = 3;
    } else if (op_name == "manual.load") {
      tensor_arg_idx = 0; tile_arg_idx = 2;
    } else if (op_name == "block.store") {
      tensor_arg_idx = 3; shapes_arg_idx = 2; tile_arg_idx = 0;
    } else if (op_name == "manual.store") {
      tensor_arg_idx = 2; tile_arg_idx = 0;
    } else if (op_name == "manual.store_fp") {
      tensor_arg_idx = 3; tile_arg_idx = 0;
    }

    if (tensor_arg_idx >= 0 &&
        static_cast<int>(op->args_.size()) > tensor_arg_idx) {
      auto tensor_var = std::dynamic_pointer_cast<const ir::Var>(op->args_[tensor_arg_idx]);

      // Select the target map based on current section
      auto& target_map = current_section_.has_value()
          ? (*current_section_ == ir::SectionKind::Cube ? cube_access_shapes_ : vec_access_shapes_)
          : access_shapes_;
      auto& target_tile_dims_map = current_section_.has_value()
          ? (*current_section_ == ir::SectionKind::Cube ? cube_access_tile_dims_ : vec_access_tile_dims_)
          : access_tile_dims_;

      if (tensor_var && target_map.find(tensor_var->name_) == target_map.end()) {
        bool found = false;
        // Try explicit shapes tuple first (block.load/store have shapes at arg[2])
        if (shapes_arg_idx >= 0 && shapes_arg_idx < static_cast<int>(op->args_.size())) {
          auto shapes_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[shapes_arg_idx]);
          if (shapes_tuple && !shapes_tuple->elements_.empty()) {
            target_map[tensor_var->name_] = shapes_tuple->elements_;
            found = true;
          }
        }
        // Fallback: infer shape from the tile type
        if (!found && tile_arg_idx >= 0 && tile_arg_idx < static_cast<int>(op->args_.size())) {
          auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(op->args_[tile_arg_idx]->GetType());
          if (tile_type) {
            target_map[tensor_var->name_] = tile_type->shape_;
          }
        }
        // Detect DN layout from manual.load kwargs
        if (tensor_var && (op_name == "manual.load" || op_name == "block.load")) {
          if (op->HasKwarg("layout") && op->GetKwarg<std::string>("layout") == "dn") {
            dn_tensors_.insert(tensor_var->name_);
          }
        }
      }
      if (tensor_var && op->HasKwarg("tile_dims") &&
          target_tile_dims_map.find(tensor_var->name_) == target_tile_dims_map.end()) {
        target_tile_dims_map[tensor_var->name_] = op->GetKwarg<std::vector<int>>("tile_dims");
      }
    }

    ir::IRVisitor::VisitExpr_(op);
  }

 private:
  std::optional<ir::SectionKind> current_section_;
};

}  // namespace

std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> CCECodegen::CollectTileVariables(
    const ir::StmtPtr& stmt) {
  if (!stmt) {
    return {};
  }

  TileCollector collector;
  collector.VisitStmt(stmt);
  return collector.tile_vars_;
}

namespace {

class BufIdVarCollector : public ir::IRVisitor {
 public:
  std::set<std::string> buf_id_var_names_;

  void VisitExpr_(const ir::CallPtr& op) override {
    if (op && op->op_) {
      const std::string& name = op->op_->name_;
      if ((name == "system.mutex_lock_dyn" || name == "system.mutex_unlock_dyn") &&
          !op->args_.empty()) {
        if (auto var = ir::As<ir::Var>(op->args_[0])) {
          buf_id_var_names_.insert(var->name_);
        }
      }
    }
    ir::IRVisitor::VisitExpr_(op);
  }
};

// Collect names of Var nodes that appear in any read position — Call args,
// Yield values, AssignStmt RHS, return values, expressions inside subscripts,
// for-loop ranges, etc. A phi return_var whose name is not in this set has
// no consumer and can be safely dropped from IfStmt codegen.
class VarReadCollector : public ir::IRVisitor {
 public:
  std::set<std::string> var_read_names_;
  std::unordered_map<std::string, int> var_read_counts_;

  // Any time we visit a Var as a sub-expression (via the default Expr walk),
  // it is in a read position — the write site is an AssignStmt.var_ which
  // the default Stmt walker does not route through VisitExpr.
  void VisitExpr_(const ir::VarPtr& op) override {
    if (op) {
      var_read_names_.insert(op->name_);
      var_read_counts_[op->name_]++;
    }
  }

  void VisitExpr_(const ir::IterArgPtr& op) override {
    if (op) {
      var_read_names_.insert(op->name_);
      var_read_counts_[op->name_]++;
      if (op->initValue_) VisitExpr(op->initValue_);
    }
  }

  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    // Skip op->var_ (LHS is a definition, not a use). Walk the RHS.
    if (op && op->value_) VisitExpr(op->value_);
  }

  void VisitStmt_(const ir::IfStmtPtr& op) override {
    // Skip op->return_vars_ — they are phi outputs (definitions), not reads.
    // Walk condition + branches explicitly.
    if (!op) return;
    if (op->condition_) VisitExpr(op->condition_);
    if (op->thenBody_) VisitStmt(op->thenBody_);
    if (op->elseBody_.has_value() && *op->elseBody_) VisitStmt(*op->elseBody_);
  }

  void VisitStmt_(const ir::ForStmtPtr& op) override {
    // Similar concern for For loops: return_vars_ (output iter args) are
    // written, not read, on each iteration's phi merge.
    if (!op) return;
    if (op->start_) VisitExpr(op->start_);
    if (op->stop_) VisitExpr(op->stop_);
    if (op->step_) VisitExpr(op->step_);
    for (const auto& ia : op->iterArgs_) {
      if (ia && ia->initValue_) VisitExpr(ia->initValue_);
    }
    if (op->body_) VisitStmt(op->body_);
  }
};

}  // namespace

void CCECodegen::CollectVarReadNames(const ir::StmtPtr& stmt,
                                     std::set<std::string>& out) const {
  if (!stmt) return;
  VarReadCollector collector;
  collector.VisitStmt(stmt);
  out = std::move(collector.var_read_names_);
  // Also populate read counts on the mutable codegen instance
  const_cast<CCECodegen*>(this)->var_read_counts_ = std::move(collector.var_read_counts_);
}

void CCECodegen::CollectBufIdVarNames(const ir::StmtPtr& stmt,
                                      std::set<std::string>& out) const {
  if (!stmt) return;
  BufIdVarCollector collector;
  collector.VisitStmt(stmt);
  out = std::move(collector.buf_id_var_names_);
}

namespace {

class MutexPipeCollector : public ir::IRVisitor {
 public:
  std::map<int, std::set<ir::PipeType>> cube_mutex_pipes;
  std::map<int, std::set<ir::PipeType>> vec_mutex_pipes;

  void VisitStmt_(const ir::SectionStmtPtr& op) override {
    auto prev = current_section_;
    current_section_ = op->sectionKind_;
    ir::IRVisitor::VisitStmt_(op);
    current_section_ = prev;
  }

  void VisitExpr_(const ir::CallPtr& op) override {
    if (op && op->op_) {
      const std::string& name = op->op_->name_;
      bool is_mutex = (name == "system.mutex_lock" || name == "system.mutex_unlock" ||
                       name == "system.mutex_lock_dyn" || name == "system.mutex_unlock_dyn");
      if (is_mutex) {
        ir::PipeType pipe = ir::PipeType::S;
        int static_bid = -1;
        std::vector<int> dyn_bids;
        for (const auto& [key, value] : op->kwargs_) {
          if (key == "pipe") pipe = static_cast<ir::PipeType>(std::any_cast<int>(value));
          if (key == "mutex_id") static_bid = std::any_cast<int>(value);
          if (key == "buf_id_values") dyn_bids = std::any_cast<std::vector<int>>(value);
        }
        auto& target = (current_section_ == ir::SectionKind::Cube)
                            ? cube_mutex_pipes : vec_mutex_pipes;
        auto record = [&](int bid) { target[bid].insert(pipe); };
        if (static_bid >= 0) record(static_bid);
        for (int bid : dyn_bids) record(bid);
        if (static_bid < 0 && dyn_bids.empty()) {
          int max_id = 2;
          for (const auto& [key, value] : op->kwargs_) {
            if (key == "max_mutex_id") max_id = std::any_cast<int>(value);
          }
          for (int i = 0; i < max_id; ++i) record(i);
        }
      }
    }
    ir::IRVisitor::VisitExpr_(op);
  }

 private:
  ir::SectionKind current_section_ = ir::SectionKind::Vector;
};

}  // namespace

void CCECodegen::CollectMutexPipeInfo(const ir::StmtPtr& stmt) {
  if (!stmt) return;
  MutexPipeCollector collector;
  collector.VisitStmt(stmt);
  cube_mutex_pipes_ = std::move(collector.cube_mutex_pipes);
  vec_mutex_pipes_ = std::move(collector.vec_mutex_pipes);
}

bool CCECodegen::ShouldSkipVPipeMutex(ir::PipeType pipe, const std::vector<int>& buf_ids) const {
  if (pipe != ir::PipeType::V || arch_ != "a5") return false;
  const auto& section_map = (current_section_kind_ == ir::SectionKind::Cube)
                                ? cube_mutex_pipes_ : vec_mutex_pipes_;
  for (int bid : buf_ids) {
    auto it = section_map.find(bid);
    if (it == section_map.end()) continue;
    for (auto p : it->second) {
      if (p != ir::PipeType::V) return false;
    }
  }
  return true;
}

std::map<std::string, std::vector<ir::ExprPtr>> CCECodegen::CollectTensorAccessShapes(
    const ir::StmtPtr& stmt) {
  if (!stmt) {
    return {};
  }

  TensorAccessShapeCollector collector;
  collector.VisitStmt(stmt);
  dn_tensors_ = collector.dn_tensors_;
  // Legacy: return the first access shape found (union of all sections)
  auto result = collector.access_shapes_;
  for (const auto& [k, v] : collector.cube_access_shapes_) {
    if (result.find(k) == result.end()) result[k] = v;
  }
  for (const auto& [k, v] : collector.vec_access_shapes_) {
    if (result.find(k) == result.end()) result[k] = v;
  }
  return result;
}

CCECodegen::SectionAccessShapes CCECodegen::CollectTensorAccessShapesPerSection(
    const ir::StmtPtr& stmt) {
  SectionAccessShapes result;
  if (!stmt) return result;

  TensorAccessShapeCollector collector;
  collector.VisitStmt(stmt);
  dn_tensors_ = collector.dn_tensors_;
  result.common_shapes = std::move(collector.access_shapes_);
  result.cube_shapes = std::move(collector.cube_access_shapes_);
  result.vec_shapes = std::move(collector.vec_access_shapes_);
  result.common_tile_dims = std::move(collector.access_tile_dims_);
  result.cube_tile_dims = std::move(collector.cube_access_tile_dims_);
  result.vec_tile_dims = std::move(collector.vec_access_tile_dims_);
  return result;
}

std::vector<int64_t> CCECodegen::ExtractShapeDimensions(const std::vector<ir::ExprPtr>& shape_exprs) {
  std::vector<int64_t> dims;
  dims.reserve(shape_exprs.size());
  for (const auto& expr : shape_exprs) {
    dims.push_back(ExtractConstInt(expr));
  }
  return dims;
}

std::string CCECodegen::FormatAddressHex(int64_t addr) {
  std::ostringstream oss;
  oss << "0x" << std::hex << addr;
  return oss.str();
}

std::string CCECodegen::GetOrCreateStructType(const std::string& fields_csv,
                                               const std::string& hint_name) {
  auto it = struct_type_defs_.find(fields_csv);
  if (it != struct_type_defs_.end()) {
    return it->second;  // Already defined �?reuse type name
  }

  // New struct type: register (definition already emitted by PreEmitStructTypes)
  std::string type_name = hint_name + "_t";
  struct_type_defs_[fields_csv] = type_name;
  return type_name;
}

namespace {
// Recursively scan IR for struct.declare calls and collect field signatures
void CollectStructDeclares(const ir::StmtPtr& stmt,
                           std::vector<std::pair<std::string, std::string>>& out) {
  if (!stmt) return;
  if (auto seq = ir::As<ir::SeqStmts>(stmt)) {
    for (const auto& s : seq->stmts_) CollectStructDeclares(s, out);
  } else if (auto eval = ir::As<ir::EvalStmt>(stmt)) {
    if (auto call = ir::As<ir::Call>(eval->expr_)) {
      if (call->op_ && call->op_->name_ == "struct.declare") {
        std::string fields = call->GetKwarg<std::string>("fields");
        std::string name = call->GetKwarg<std::string>("array");
        out.emplace_back(fields, name);
      }
    }
  } else if (auto for_stmt = ir::As<ir::ForStmt>(stmt)) {
    CollectStructDeclares(for_stmt->body_, out);
  } else if (auto if_stmt = ir::As<ir::IfStmt>(stmt)) {
    CollectStructDeclares(if_stmt->thenBody_, out);
    if (if_stmt->elseBody_) CollectStructDeclares(*if_stmt->elseBody_, out);
  } else if (auto section = ir::As<ir::SectionStmt>(stmt)) {
    CollectStructDeclares(section->body_, out);
  }
}
}  // namespace

void CCECodegen::PreEmitStructTypes(const ir::StmtPtr& body) {
  std::vector<std::pair<std::string, std::string>> declares;
  CollectStructDeclares(body, declares);
  // Track used type names to avoid collision when different field sets share a hint name
  std::set<std::string> used_type_names;
  for (const auto& [fields_csv, hint_name] : declares) {
    if (struct_type_defs_.count(fields_csv)) continue;
    std::string type_name = hint_name + "_t";
    // Ensure uniqueness: if type_name is taken by a different field set, add suffix
    if (used_type_names.count(type_name)) {
      int suffix = 1;
      while (used_type_names.count(type_name + "_" + std::to_string(suffix))) suffix++;
      type_name = type_name + "_" + std::to_string(suffix);
    }
    struct_type_defs_[fields_csv] = type_name;
    used_type_names.insert(type_name);

    // Parse fields and emit struct definition
    std::vector<std::string> field_names;
    std::istringstream iss(fields_csv);
    std::string token;
    while (std::getline(iss, token, ',')) {
      if (!token.empty()) field_names.push_back(token);
    }
    std::string def = "struct " + type_name + " { ";
    for (const auto& f : field_names) {
      def += "int64_t " + f + "; ";
    }
    def += "};";
    emitter_.EmitLine(def);
  }
}

void CCECodegen::GenerateTileTypeDeclaration(const std::string& var_name, const ir::TileTypePtr& tile_type) {
  INTERNAL_CHECK(!var_name.empty()) << "Internal error: var_name cannot be empty";
  INTERNAL_CHECK(tile_type != nullptr) << "Internal error: tile_type is null";

  // Extract tile shape dimensions
  std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);

  // CCE codegen only supports 1D and 2D tiles
  CHECK(shape_dims.size() <= 2) << "CCE codegen only supports 1D and 2D TileType, but got "
                                << shape_dims.size()
                                << " dimensions. Multi-dimensional tiles (>2D) are supported at IR level "
                                << "but not yet in code generation.";

  // Determine tile dimensions (default to 1 if not specified)
  int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
  int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;

  // Extract valid_shape: compute runtime ctor args.
  auto vs = ExtractValidShapeInfo(tile_type, rows, cols,
                                  [this](const ir::VarPtr& v) { return GetVarName(v); });
  std::string ctor_args = BuildTileCtorArgs(vs, rows, cols);

  // Generate Tile type alias (with dedup: reuse alias if same type string already emitted)
  std::string tile_type_str = type_converter_.ConvertTileType(tile_type, rows, cols);
  std::string type_alias_name;
  auto dedup_it = emitted_tile_types_.find(tile_type_str);
  if (dedup_it != emitted_tile_types_.end()) {
    type_alias_name = dedup_it->second;
  } else {
    // Derive alias name from var_name base (strip trailing _N suffix for dedup readability)
    auto last_underscore = var_name.rfind('_');
    bool has_index_suffix = (last_underscore != std::string::npos &&
                             last_underscore + 1 < var_name.size() &&
                             std::isdigit(var_name[last_underscore + 1]));
    std::string base_name = has_index_suffix ? var_name.substr(0, last_underscore) : var_name;
    type_alias_name = base_name + "_Type";
    // Ensure uniqueness against existing aliases
    if (emitted_tile_types_.count(type_alias_name + "_used") > 0 &&
        emitted_tile_types_[type_alias_name + "_used"] != tile_type_str) {
      type_alias_name = var_name + "Type";
    }
    emitted_tile_types_[tile_type_str] = type_alias_name;
    emitted_tile_types_[type_alias_name + "_used"] = tile_type_str;
    emitter_.EmitLine("using " + type_alias_name + " = " + tile_type_str + ";");
  }

  // Generate Tile instance + TASSIGN on one line (compact)
  // Only pass ctor args when template has -1 params (dynamic valid_shape).
  std::string ctor_suffix = vs.needs_ctor ? ("(" + ctor_args + ")") : "";
  if (tile_type->memref_.has_value()) {
    int64_t addr =
        ExtractConstInt((*tile_type->memref_)->addr_);  // NOLINT(bugprone-unchecked-optional-access)
    std::string addr_str = FormatAddressHex(addr);
    emitter_.EmitLine(type_alias_name + " " + var_name + ctor_suffix +
                      "; TASSIGN(" + var_name + ", " + addr_str + ");");
    tile_addresses_[var_name] = addr_str;
  } else {
    emitter_.EmitLine(type_alias_name + " " + var_name + ctor_suffix + ";");
  }
}

// ========================================================================
// Phase 7 helpers: Stride type generation and GlobalTensor instance emission
// ========================================================================

std::string CCECodegen::GenerateSingleFileStrideType(const std::vector<int64_t>& shape_dims,
                                                     const std::vector<int64_t>& tensor_dims,
                                                     bool all_static,
                                                     bool needs_dynamic_stride,
                                                     bool needs_tile_dims_stride) const {
  std::ostringstream oss;
  oss << "pto::Stride<";
  const size_t target_dims = 5;
  size_t n = shape_dims.size();

  if ((needs_dynamic_stride || needs_tile_dims_stride) && n == 2) {
    // Dynamic tensor with access window: allow all five stride slots to be
    // provided at runtime. This is required for tile_dims=[1, 3] BSND views,
    // where the row stride is N*D/N*Skv instead of the access-window column.
    for (size_t i = 0; i < target_dims; ++i) {
      oss << "-1";
      if (i < target_dims - 1) oss << ", ";
    }
  } else if (force_dn_layout_ && n == 2) {
    // DN (column-major) strides for 2D: [1, dim0] (column-stride=1, row-stride=dim0)
    for (size_t i = 0; i < target_dims - n; ++i) {
      oss << shape_dims[0] << ", ";
    }
    oss << "1, " << shape_dims[0];
  } else {
    // ND (row-major) strides: stride[i] = product(dims[i+1..n-1])
    // When access_shape overrides Shape<> for sub-tile operations (e.g., l0c_store of [128,128]
    // into a larger [12288,1024] tensor), use actual tensor dims for stride so the row spacing
    // matches the real tensor layout, not the sub-tile size.
    const auto& stride_source = (all_static && !tensor_dims.empty()) ? tensor_dims : shape_dims;
    for (size_t i = 0; i < target_dims - n; ++i) {
      oss << "1, ";
    }
    for (size_t i = 0; i < n; ++i) {
      int64_t stride = 1;
      for (size_t j = i + 1; j < n; ++j) {
        stride *= stride_source[j];
      }
      oss << stride;
      if (i < n - 1) oss << ", ";
    }
  }
  oss << ">";
  return oss.str();
}

void CCECodegen::EmitGlobalTensorInstance(const std::string& var_name,
                                          const std::string& global_type_name,
                                          const std::string& shape_type_name,
                                          const std::string& stride_type_name,
                                          const ir::TensorTypePtr& tensor_type,
                                          const std::vector<int64_t>& shape_dims,
                                          bool needs_dynamic_stride,
                                          bool needs_tile_dims_stride,
                                          const std::optional<std::vector<int>>& tile_dims,
                                          const std::optional<std::string>& base_pointer,
                                          const std::optional<std::string>& tensor_struct_ptr,
                                          bool use_runtime_tensor_struct) {
  std::ostringstream global_instance;
  global_instance << global_type_name << " " << var_name << "(";
  if (base_pointer.has_value()) {
    global_instance << base_pointer.value();
  }
  if ((needs_dynamic_stride || needs_tile_dims_stride) && base_pointer.has_value()) {
    // Dynamic stride: tensor has dynamic dims but access_shape overrides Shape<>.
    // The row stride must use the tensor's actual last dim (e.g., Skv), not access shape col (e.g., 128).
    auto dim_expr = [this](const ir::ExprPtr& dim, int64_t fallback) -> std::string {
      if (auto ci = std::dynamic_pointer_cast<const ir::ConstInt>(dim)) {
        return std::to_string(ci->value_);
      }
      if (auto var = std::dynamic_pointer_cast<const ir::Var>(dim)) {
        return context_.GetVarName(std::const_pointer_cast<ir::Var>(var));
      }
      return std::to_string(fallback);
    };
    auto full_stride_expr = [&](int dim_idx) -> std::string {
      std::string expr = "1";
      for (size_t i = static_cast<size_t>(dim_idx) + 1; i < tensor_type->shape_.size(); ++i) {
        std::string dim = dim_expr(tensor_type->shape_[i], 1);
        expr = (expr == "1") ? dim : expr + "*" + dim;
      }
      return expr;
    };

    std::string row_stride_expr;
    std::string col_stride_expr;
    if (needs_tile_dims_stride && tile_dims.has_value() && tile_dims->size() == 2) {
      row_stride_expr = full_stride_expr((*tile_dims)[0]);
      col_stride_expr = full_stride_expr((*tile_dims)[1]);
    } else {
      row_stride_expr = dim_expr(tensor_type->shape_.back(), shape_dims.back());
      col_stride_expr = "1";
    }
    std::string leading_stride = std::to_string(shape_dims[0]) + "*" + row_stride_expr;
    global_instance << ", " << shape_type_name << "(), " << stride_type_name << "(";
    if (force_dn_layout_) {
      global_instance << leading_stride << ", " << leading_stride << ", " << leading_stride
                      << ", " << col_stride_expr << ", " << row_stride_expr;
    } else {
      global_instance << leading_stride << ", " << leading_stride << ", " << leading_stride
                      << ", " << row_stride_expr << ", " << col_stride_expr;
    }
    global_instance << ")";
  } else if (use_runtime_tensor_struct && tensor_struct_ptr.has_value()) {
    // Use original tensor ndim for stride indices (strides come from the full tensor struct)
    size_t ndim = tensor_type->shape_.size();
    global_instance << ", {}, {";
    for (size_t i = 0; i < ndim; i++) {
      global_instance << "static_cast<int64_t>(" << tensor_struct_ptr.value() << "->strides["
                      << std::to_string(i) << "])";
      if (i != ndim - 1) {
        global_instance << ", ";
      }
    }
    global_instance << "}";
  }
  global_instance << ");";
  emitter_.EmitLine(global_instance.str());

  // Register pointer mapping if base_pointer provided
  if (base_pointer.has_value()) {
    context_.RegisterPointer(var_name, base_pointer.value());
  }

  // Register both pointer and Tensor struct mappings if provided
  if (tensor_struct_ptr.has_value()) {
    // Register Tensor struct pointer for stride access
    context_.RegisterTensorStruct(var_name, tensor_struct_ptr.value());
  }
}

void CCECodegen::GenerateGlobalTensorTypeDeclaration(
    const std::string& var_name, const ir::TensorTypePtr& tensor_type,
    const std::optional<std::string>& base_pointer, const std::optional<std::string>& tensor_struct_ptr,
    const std::optional<std::vector<ir::ExprPtr>>& access_shape) {
  INTERNAL_CHECK(!var_name.empty()) << "Internal error: var_name cannot be empty";
  INTERNAL_CHECK(tensor_type != nullptr) << "Internal error: tensor_type is null";

  // Check if all tensor shape dims are static (ConstInt)
  bool all_static = true;
  for (const auto& dim : tensor_type->shape_) {
    if (!std::dynamic_pointer_cast<const ir::ConstInt>(dim)) {
      all_static = false;
      break;
    }
  }

  // Full tensor dimensions (only needed when tensor_struct_ptr is provided, for stride args)
  std::vector<int64_t> tensor_dims;
  if (all_static) {
    tensor_dims = ExtractShapeDimensions(tensor_type->shape_);
  }

  // Access shape overrides Shape<>/Stride<> type parameters when available
  std::vector<int64_t> shape_dims;
  if (access_shape.has_value()) {
    shape_dims = ExtractShapeDimensions(access_shape.value());
  } else if (all_static) {
    shape_dims = tensor_dims;
  } else {
    // Dynamic shapes without access_shape: use [1, 1] as a minimal placeholder
    // (the actual address is set by TASSIGN at runtime, shape only affects type template)
    shape_dims = {1, 1};
  }

  // Get element type
  std::string element_type = tensor_type->dtype_.ToCTypeString();
  const bool is_nz_layout = IsNZTensorType(tensor_type);

  // Generate unique type names for this variable
  std::string shape_type_name = var_name + "ShapeDim5";
  std::string stride_type_name = var_name + "StrideDim5";
  std::string global_type_name = var_name + "Type";

  std::vector<int64_t> emitted_shape_dims = shape_dims;
  if (is_nz_layout) {
    if (!all_static) {
      throw pypto::ir::ValueError("CCE NZ tensor lowering currently requires static tensor shapes");
    }
    emitted_shape_dims = BuildNZPhysicalShapeDims(tensor_type, tensor_dims);
  }

  // Generate Shape type alias
  std::string shape_type = type_converter_.GenerateShapeType(emitted_shape_dims);
  std::ostringstream shape_alias;
  shape_alias << "using " << shape_type_name << " = " << shape_type << ";";
  emitter_.EmitLine(shape_alias.str());

  // Generate Stride type alias
  bool needs_tile_dims_stride = !all_static && access_shape.has_value() &&
                                current_tile_dims_.has_value() &&
                                current_tile_dims_->size() == 2 && !is_nz_layout;
  bool needs_dynamic_stride = !all_static && access_shape.has_value() && !force_dn_layout_ && !is_nz_layout;
  std::string stride_type;
  if (is_nz_layout) {
    stride_type = GenerateSingleFileStrideType(emitted_shape_dims, emitted_shape_dims, true, false, false);
  } else if (single_file_mode_) {
    stride_type = GenerateSingleFileStrideType(shape_dims, tensor_dims, all_static,
                                               needs_dynamic_stride, needs_tile_dims_stride);
  } else {
    stride_type = type_converter_.GenerateStrideType(shape_dims);
  }
  std::ostringstream stride_alias;
  stride_alias << "using " << stride_type_name << " = " << stride_type << ";";
  emitter_.EmitLine(stride_alias.str());

  // Determine layout: DN if last dim is 1 or if tensor is in dn_tensors_ set
  std::string global_layout_arg = stride_type_name;
  if (is_nz_layout) {
    global_layout_arg += ", Layout::NZ";
  } else if (*shape_dims.rbegin() == 1 || force_dn_layout_) {
    global_layout_arg += ", Layout::DN";
  }

  // Generate GlobalTensor type alias
  std::ostringstream global_type_alias;
  global_type_alias << "using " << global_type_name << " = GlobalTensor<" << element_type << ", "
                    << shape_type_name << ", " << global_layout_arg << ">;";
  emitter_.EmitLine(global_type_alias.str());

  // Generate GlobalTensor instance and register mappings
  EmitGlobalTensorInstance(var_name, global_type_name, shape_type_name, stride_type_name, tensor_type,
                           emitted_shape_dims, needs_dynamic_stride, needs_tile_dims_stride, current_tile_dims_,
                           base_pointer, tensor_struct_ptr, tensor_struct_ptr.has_value() && !is_nz_layout);
}

}  // namespace codegen

}  // namespace pypto
