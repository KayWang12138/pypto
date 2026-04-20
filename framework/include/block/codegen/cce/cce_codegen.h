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

#ifndef PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_
#define PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "block/backend/common/backend.h"
#include "block/codegen/cce/code_context.h"
#include "block/codegen/cce/code_emitter.h"
#include "block/codegen/cce/type_converter.h"
#include "block/codegen/codegen_base.h"
#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {

namespace codegen {

/**
 * @brief CCE code generator for converting PyPTO IR to pto-isa C++ code
 *
 * CCECodegen traverses the IR using the visitor pattern and generates
 * compilable C++ code using pto-isa instructions. It handles:
 * - Function prologue (signature, argument unpacking, type definitions)
 * - Function body (block operations, sync operations, control flow)
 * - Type conversions and memory management
 */
class CCECodegen : public CodegenBase {
 protected:
    using CodegenBase::VisitStmt_;
    using CodegenBase::VisitExpr_;

 public:
  /** @brief Default constructor (backend is always CCE) */
  CCECodegen();

  /**
   * @brief Generate C++ code from a PyPTO IR Program
   *
   * Classifies functions into kernel and orchestration, then generates:
   * - Kernel functions -> kernels/<func_name>.cpp (CCE kernel C++ code)
   * - Orchestration function -> orchestration/<func_name>.cpp (orchestration C++ code)
   *
   * @param program The IR Program to generate code for
   * @return Map from file path to generated C++ code content
   */
  [[nodiscard]] std::map<std::string, std::string> Generate(const ir::ProgramPtr& program);

  /**
   * @brief Generate a single C++ file from a PyPTO IR Program (MIX mode)
   *
   * Runs IR passes (LowerBreakContinue → ConvertToSSA → ConstFoldAndSimplify),
   * then generates a single __global__ AICORE kernel with:
   * - PTO-style function signature
   * - Section-aware tile declarations (#if __DAV_CUBE__ / __DAV_VEC__)
   * - constexpr for compile-time constants
   * - FFTS support for cross-core sync
   *
   * @param program The IR Program to generate code for
   * @return Generated C++ code as a single string
   */
  [[nodiscard]] std::string GenerateSingle(const ir::ProgramPtr& program,
                                            const std::string& arch = "a3");

  // CodegenBase interface (unified API for operator codegen callbacks)
  [[nodiscard]] std::string GetCurrentResultTarget() const override { return current_target_var_; }
  void Emit(const std::string& line) override;
  std::string GetExprAsCode(const ir::ExprPtr& expr) override;
  [[nodiscard]] std::string GetTypeString(const ir::DataType& dtype) const override;
  void set_in_vf_scope(bool v) { in_vf_scope_ = v; if (!v) { vf_ptr_vars_.clear(); vf_post_update_ptrs_.clear(); } }
  [[nodiscard]] bool in_vf_scope() const { return in_vf_scope_; }
  int GetTileOffsetCounter() { return tile_offset_counter_++; }
  void RegisterVFPtrVar(const std::string& name) { vf_ptr_vars_.insert(name); }
  [[nodiscard]] bool IsVFPtrVar(const std::string& name) const { return vf_ptr_vars_.count(name) > 0; }
  std::string GetOrCreatePostUpdatePtr(const std::string& key, const std::string& ptr_type,
                                       const std::string& init_expr) {
    auto it = vf_post_update_ptrs_.find(key);
    if (it != vf_post_update_ptrs_.end()) return it->second;
    std::string var = "_vf_st_ptr_" + std::to_string(GetTileOffsetCounter());
    if (loop_depth_ > 0) {
      // Inside loop: hoist declaration before the loop
      loop_hoisted_decls_.push_back("__ubuf__ " + ptr_type + " *" + var + " = " + init_expr + ";");
    } else {
      // Not in loop (single iteration or no loop): emit directly
      Emit("__ubuf__ " + ptr_type + " *" + var + " = " + init_expr + ";");
    }
    vf_post_update_ptrs_[key] = var;
    return var;
  }
  int64_t GetConstIntValue(const ir::ExprPtr& expr) override;
  std::string GetVarName(const ir::VarPtr& var) override;

  const TypeConverter& GetTypeConverter() const { return type_converter_; }

  /** @brief Check if currently generating in single-file MIX mode */
  bool IsSingleFileMode() const { return single_file_mode_; }

  /** @brief Get the target architecture */
  const std::string& GetArch() const { return arch_; }

  /** @brief Get current section kind (Cube or Vector), nullopt if not in any section */
  std::optional<ir::SectionKind> GetCurrentSectionKind() const { return current_section_kind_; }

  /** @brief Check if currently generating code inside a Cube section */
  bool IsInCubeSection() const {
    return current_section_kind_.has_value() && *current_section_kind_ == ir::SectionKind::Cube;
  }

  /** @brief Get the base address of a tile variable (from TASSIGN in prologue) */
  std::string GetTileAddress(const std::string& tile_name) const {
    auto it = tile_addresses_.find(tile_name);
    if (it != tile_addresses_.end()) return it->second;
    return "0x0";
  }

  /**
   * @brief Compute offset from IR tensor shape (for single-file mode without Tensor struct)
   *
   * Computes row-major stride-based offset: off[0]*stride[0] + off[1]*stride[1] + ...
   * where stride[i] = product(shape[i+1..n-1])
   */
  std::string ComputeIRBasedOffset(const ir::TensorTypePtr& tensor_type,
                                   const ir::MakeTuplePtr& offsets);

  /**
   * @brief Get pointer name for a variable (CCE-specific)
   */
  std::string GetPointer(const std::string& var_name);

  /**
   * @brief Register pointer mapping for block.store result (CCE-specific)
   *
   * Associates the assignment target variable with the output tensor variable
   * for pointer lookup. Used when block.store returns a tensor reference.
   *
   * @param output_var_name Assignment target variable name
   * @param tensor_var_name Output tensor variable name (e.g., from GlobalTensor)
   */
  void RegisterOutputPointer(const std::string& output_var_name, const std::string& tensor_var_name);

  /**
   * @brief Get Tensor struct pointer name for a variable (CCE-specific)
   */
  std::string GetTensorStruct(const std::string& var_name);

  /**
   * @brief Register Tensor struct mapping for block.store result (CCE-specific)
   *
   * Associates the assignment target variable with the output tensor variable
   * for Tensor struct lookup. Used when block.store returns a tensor reference.
   *
   * @param output_var_name Assignment target variable name
   * @param tensor_var_name Output tensor variable name (e.g., from GlobalTensor)
   */
  void RegisterOutputTensorStruct(const std::string& output_var_name, const std::string& tensor_var_name);

  /**
   * @brief Get or create a C++ struct type for the given field signature.
   * Deduplicates identical structs: same fields → same type name.
   */
  std::string GetOrCreateStructType(const std::string& fields_csv, const std::string& hint_name);

  /**
   * @brief Check whether a PIPE_V mutex lock/unlock should be skipped.
   *
   * Returns true (skip) when arch is A5, pipe is PIPE_V, and all buf_ids
   * in the vector are only used by PIPE_V (no cross-pipe synchronization needed).
   */
  bool ShouldSkipVPipeMutex(ir::PipeType pipe, const std::vector<int>& buf_ids) const;

 protected:
  // Override visitor methods for code generation - Statements
  void VisitStmt_(const ir::AssignStmtPtr& op) override;
  void VisitStmt_(const ir::EvalStmtPtr& op) override;
  void VisitStmt_(const ir::ReturnStmtPtr& op) override;
  void VisitStmt_(const ir::ForStmtPtr& op) override;
  void VisitStmt_(const ir::WhileStmtPtr& op) override;
  void VisitStmt_(const ir::IfStmtPtr& op) override;
  void VisitStmt_(const ir::YieldStmtPtr& op) override;
  void VisitStmt_(const ir::SectionStmtPtr& op) override;

  // Override visitor methods for code generation - Expressions
  // Leaf nodes
  void VisitExpr_(const ir::VarPtr& op) override;
  void VisitExpr_(const ir::IterArgPtr& op) override;
  void VisitExpr_(const ir::ConstIntPtr& op) override;
  void VisitExpr_(const ir::ConstFloatPtr& op) override;
  void VisitExpr_(const ir::ConstBoolPtr& op) override;
  void VisitExpr_(const ir::CallPtr& op) override;
  void VisitExpr_(const ir::TupleGetItemExprPtr& op) override;
  void VisitExpr_(const ir::TileOffsetExprPtr& op) override;

  // Binary operations
  void VisitExpr_(const ir::AddPtr& op) override;
  void VisitExpr_(const ir::SubPtr& op) override;
  void VisitExpr_(const ir::MulPtr& op) override;
  void VisitExpr_(const ir::FloorDivPtr& op) override;
  void VisitExpr_(const ir::FloorModPtr& op) override;
  void VisitExpr_(const ir::FloatDivPtr& op) override;
  void VisitExpr_(const ir::MinPtr& op) override;
  void VisitExpr_(const ir::MaxPtr& op) override;
  void VisitExpr_(const ir::PowPtr& op) override;
  void VisitExpr_(const ir::EqPtr& op) override;
  void VisitExpr_(const ir::NePtr& op) override;
  void VisitExpr_(const ir::LtPtr& op) override;
  void VisitExpr_(const ir::LePtr& op) override;
  void VisitExpr_(const ir::GtPtr& op) override;
  void VisitExpr_(const ir::GePtr& op) override;
  void VisitExpr_(const ir::AndPtr& op) override;
  void VisitExpr_(const ir::OrPtr& op) override;
  void VisitExpr_(const ir::XorPtr& op) override;
  void VisitExpr_(const ir::BitAndPtr& op) override;
  void VisitExpr_(const ir::BitOrPtr& op) override;
  void VisitExpr_(const ir::BitXorPtr& op) override;
  void VisitExpr_(const ir::BitShiftLeftPtr& op) override;
  void VisitExpr_(const ir::BitShiftRightPtr& op) override;

  // Unary operations
  void VisitExpr_(const ir::AbsPtr& op) override;
  void VisitExpr_(const ir::NegPtr& op) override;
  void VisitExpr_(const ir::NotPtr& op) override;
  void VisitExpr_(const ir::BitNotPtr& op) override;
  void VisitExpr_(const ir::CastPtr& op) override;

 private:
  /**
   * @brief Extract yield variable names from an IR body without emitting code.
   *
   * Inspects the trailing YieldStmt in @p body (which may be a bare YieldStmt
   * or the last statement of a SeqStmts) and returns the resolved name for each
   * yielded value.  Handles Var, ConstInt, IterArg, and TupleGetItemExpr
   * (including inlined MakeTuple).  Returns an empty vector when any yield
   * value cannot be resolved.
   */
  std::vector<std::string> ExtractYieldNames(const ir::StmtPtr& body) const;

  /**
   * @brief Emit yield-assignment code for if-stmt return variables.
   *
   * For each return variable, resolves the corresponding yield value from
   * @p target_names, emits a TASSIGN (Tile/Tuple) or scalar assignment, and
   * propagates pointer / tensor-struct mappings.  Clears yield_buffer_ when
   * done.
   */
  void EmitYieldAssignments(const std::vector<ir::VarPtr>& return_vars,
                            const std::vector<std::string>& target_names);

  /**
   * @brief Try to emit optimised N-way select (array + index) for an IfStmt.
   *
   * Detects nested if-else chains of the form
   *   if (x == 0) { yield A } else { if (x == 1) { yield B } ... }
   * and lowers them into a constexpr array indexed by x.
   *
   * @return true if the optimisation was applied (caller should return).
   */
  bool TryEmitNWaySelect(const ir::IfStmtPtr& op);

  /**
   * @brief Try to emit the identity-else optimised IfStmt codegen.
   *
   * When the else branch is a pure yield whose values are already live before
   * the if, we can skip the else branch entirely and emit assignments only in
   * the then branch.
   *
   * @return true if identity-else was detected and emitted.
   */
  bool TryEmitIdentityElseIf(const ir::IfStmtPtr& op);

  /**
   * @brief Emit full phi-style IfStmt codegen.
   *
   * Declares return variables before the if, emits then and else bodies, and
   * writes yield-assignment code in each branch.
   */
  void EmitFullPhiIf(const ir::IfStmtPtr& op);

  /**
   * @brief Generate function prologue
   *
   * Emits function signature, argument unpacking, GlobalTensor declarations,
   * and Tile declarations with TASSIGN.
   *
   * @param func The function to generate prologue for
   */
  void GeneratePrologue(const ir::FunctionPtr& func);

  /**
   * @brief Generate function body
   *
   * Visits the function body statement to generate the main code.
   *
   * @param func The function to generate body for
   */
  void GenerateBody(const ir::FunctionPtr& func);

  /**
   * @brief Extract constant integer value from expression
   *
   * @param expr The expression (must be ConstInt)
   * @return The integer value
   */
  int64_t ExtractConstInt(const ir::ExprPtr& expr);

  /**
   * @brief Collect all TileType variables from function body
   *
   * Recursively traverses the statement tree to find all variables
   * with TileType that need Tile declarations in the prologue.
   *
   * @param stmt The statement to scan (typically func->body_)
   * @return Vector of (Var, TileType) pairs
   */
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> CollectTileVariables(const ir::StmtPtr& stmt);

  /**
   * @brief Collect tensor access window shapes from block.load/store operations
   *
   * Scans the function body for block.load/block.store calls
   * and extracts the shapes_tuple for each tensor parameter. The GlobalTensor
   * Shape<> should use this access window shape, not the full tensor shape.
   *
   * @param stmt The statement to scan (typically func->body_)
   * @return Map from tensor VarPtr to access window shape expressions
   */
  std::map<std::string, std::vector<ir::ExprPtr>> CollectTensorAccessShapes(const ir::StmtPtr& stmt);

  /// Per-section access shapes for GlobalTensor declarations
  struct SectionAccessShapes {
    std::map<std::string, std::vector<ir::ExprPtr>> common_shapes;  // outside any section
    std::map<std::string, std::vector<ir::ExprPtr>> cube_shapes;
    std::map<std::string, std::vector<ir::ExprPtr>> vec_shapes;
  };

  SectionAccessShapes CollectTensorAccessShapesPerSection(const ir::StmtPtr& stmt);

  /**
   * @brief Extract shape dimensions from shape expressions
   *
   * Converts a vector of shape expressions (assumed to be ConstInt)
   * into a vector of integer dimensions.
   *
   * @param shape_exprs Vector of shape expressions (ConstInt)
   * @return Vector of integer dimensions
   */
  std::vector<int64_t> ExtractShapeDimensions(const std::vector<ir::ExprPtr>& shape_exprs);

  /**
   * @brief Format address as hexadecimal string
   *
   * Converts an integer address to hex format for TASSIGN instructions.
   *
   * @param addr Address value
   * @return Hex string (e.g., "0x0", "0x10000")
   */
  std::string FormatAddressHex(int64_t addr);

  /**
   * @brief Get or create a C++ struct type for the given field signature.
   *
   * Deduplicates identical structs: same fields → same type name.
   * The struct type definition is emitted once (on first encounter).
   *
   * @param fields_csv Comma-separated field names (dedup key)
   * @param hint_name Name hint for the type (used if this is the first struct with these fields)
   * @return The canonical type name (e.g., "ctx_t")
   */
  void PreEmitStructTypes(const ir::StmtPtr& body);

  /**
   * @brief Generate CCE kernel C++ code for a single function
   *
   * Emits function prologue (signature, argument unpacking, type declarations)
   * and body (block operations, control flow) for kernel (InCore) functions.
   *
   * @param func The kernel function to generate code for
   * @return Generated C++ code as a string
   */
  std::string GenerateFunction(const ir::FunctionPtr& func);

  /**
   * @brief Generate config file for orchestration and kernels
   *
   * @param orch_func_name Orchestration function name
   * @param func_name_to_id Kernel function name -> func id mapping
   * @param func_name_to_core_type Kernel function name -> core type mapping
   * @return Generated config file as a string
   */
  std::string GenerateConfigFile(const std::string& orch_func_name,
                                 const std::map<std::string, int>& func_name_to_id,
                                 const std::map<std::string, ir::CoreType>& func_name_to_core_type);

  /**
   * @brief Generate Tile type declaration and instance
   *
   * Emits type alias and instance declaration for a Tile variable.
   * Automatically extracts memref address from tile_type if present and emits TASSIGN.
   *
   * @param var_name Variable name for the tile
   * @param tile_type The TileType to generate declaration for (memref extracted automatically)
   */
  void GenerateTileTypeDeclaration(const std::string& var_name, const ir::TileTypePtr& tile_type);

  /**
   * @brief Generate GlobalTensor type declaration and instance
   *
   * Emits shape type alias, stride type alias, GlobalTensor type alias,
   * and instance declaration for a GlobalTensor variable.
   *
   * @param var_name Variable name for the global tensor
   * @param tensor_type The TensorType to generate declaration for
   * @param base_pointer Optional base pointer name for initialization
   * @param tensor_struct_ptr Optional Tensor struct pointer name for initialization
   * @param access_shape Optional access window shape from block.load/store (overrides tensor shape for
   * Shape<>/Stride<>)
   */
  void GenerateGlobalTensorTypeDeclaration(
      const std::string& var_name, const ir::TensorTypePtr& tensor_type,
      const std::optional<std::string>& base_pointer = std::nullopt,
      const std::optional<std::string>& tensor_struct_ptr = std::nullopt,
      const std::optional<std::vector<ir::ExprPtr>>& access_shape = std::nullopt);

  /**
   * @brief Generate PTO-style function signature and prologue for single-file mode
   *
   * Emits __global__ AICORE void func_name(__gm__ type* p, ...) with constexpr
   * scalars and section-aware tile declarations.
   */
  void GenerateSinglePrologue(const ir::FunctionPtr& func, bool has_cross_sync);

  /**
   * @brief Detect whether the program uses cross-core sync ops
   */
  bool DetectCrossCoreSyncOps(const ir::StmtPtr& stmt);

  /**
   * @brief Collect which section each tile belongs to (for section-aware declaration)
   */
  std::map<ir::VarPtr, ir::SectionKind> CollectTileSections(const ir::StmtPtr& stmt);

  // --- Phase 5: ForStmt helpers ---

  /**
   * @brief Register iteration arguments for a for-loop and emit their initialization.
   *
   * Handles alias propagation, cross-section safety, and pointer/struct inheritance.
   * Returns the sanitized names of each iter-arg for later yield assignment.
   */
  std::vector<std::string> RegisterForIterArgs(const ir::ForStmtPtr& op);

  /**
   * @brief Emit yield-to-iter-arg assignments at the end of a for-loop body.
   *
   * Resolves aliases, detects self-assignments, and uses TASSIGN for tile transfers.
   */
  void EmitForYieldAssignments(const std::vector<std::string>& iter_arg_names);

  void EmitForLoopWithHoisting(const ir::ForStmtPtr& op,
                               const std::string& loop_var_name,
                               const std::vector<std::string>& iter_arg_names,
                               const std::string& start,
                               const std::string& stop,
                               const std::string& step);

  // --- Phase 6: GenerateSinglePrologue helpers ---

  /**
   * @brief Emit the __global__ AICORE function signature and opening boilerplate.
   *
   * Collects dynamic dim variables, builds the parameter list, emits the opening
   * brace, registers dynamic dims, and emits FFTS setup if needed.
   */
  void EmitSingleFunctionSignature(const ir::FunctionPtr& func, bool has_cross_sync);

  /**
   * @brief Emit section-aware GlobalTensor declarations for single-file mode.
   */
  void EmitSingleGlobalTensors(const ir::FunctionPtr& func, const SectionAccessShapes& section_shapes);

  /**
   * @brief Emit section-aware Tile declarations for single-file mode.
   */
  void EmitSingleTileDeclarations(const ir::FunctionPtr& func);

  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> FilterPrologueTiles(
      const ir::FunctionPtr& func,
      std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>>& all_tiles_out,
      std::vector<std::pair<ir::VarPtr, ir::VarPtr>>& deduped_aliases_out);

  void EmitSectionAwareTiles(
      const std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>>& tile_vars,
      const std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>>& all_tiles,
      const std::vector<std::pair<ir::VarPtr, ir::VarPtr>>& deduped_aliases,
      const std::map<ir::VarPtr, ir::SectionKind>& tile_sections,
      const std::map<std::string, std::set<ir::SectionKind>>& tile_usage_sections);

  // --- Phase 7: GenerateGlobalTensorTypeDeclaration helpers ---

  /**
   * @brief Generate the stride type string for single-file mode.
   *
   * Handles ND, DN, and dynamic stride layouts for the pto::Stride<> template.
   */
  std::string GenerateSingleFileStrideType(const std::vector<int64_t>& shape_dims,
                                           const std::vector<int64_t>& tensor_dims,
                                           bool all_static, bool needs_dynamic_stride) const;

  /**
   * @brief Emit the GlobalTensor instance declaration and register pointer/struct mappings.
   */
  void EmitGlobalTensorInstance(const std::string& var_name,
                                const std::string& global_type_name,
                                const std::string& shape_type_name,
                                const std::string& stride_type_name,
                                const ir::TensorTypePtr& tensor_type,
                                const std::vector<int64_t>& shape_dims,
                                bool needs_dynamic_stride,
                                const std::optional<std::string>& base_pointer,
                                const std::optional<std::string>& tensor_struct_ptr,
                                bool use_runtime_tensor_struct);

  // --- Phase 8: AssignStmt helper ---

  /**
   * @brief Handle tile-related TupleGetItemExpr assignments (early return path).
   *
   * Returns true if the assignment was handled (caller should return early).
   */
  bool HandleTileRelatedAssignment(const ir::AssignStmtPtr& op);

  // --- Phase 9: GeneratePrologue helper ---

  /**
   * @brief Unpack function arguments from the args array and emit type declarations.
   */
  void UnpackFunctionArguments(const ir::FunctionPtr& func,
                               const std::map<std::string, std::vector<ir::ExprPtr>>& access_shapes);

  // Dual-mode context for expression visitor pattern
  std::string current_target_var_;         ///< INPUT: Assignment target variable name (for Call expressions)
  std::string current_expr_value_;         ///< OUTPUT: Inline C++ value for scalar expressions
  std::vector<std::string> yield_buffer_;  ///< Temporary storage for yielded values from loops

  CodeEmitter emitter_;              ///< Code emitter for structured output
  CodeContext context_;              ///< Context for variable tracking
  TypeConverter type_converter_;     ///< Type converter
  const backend::Backend* backend_;  ///< CCE backend instance (for op info, core type, orchestration)
  bool single_file_mode_ = false;    ///< Whether generating in single-file MIX mode
  std::string arch_ = "a3";          ///< Target architecture ("a2", "a3", "a5")
  std::optional<ir::SectionKind> current_section_kind_;  ///< Current section being generated (Cube/Vector)
  bool force_dn_layout_ = false;     ///< Temporary flag for DN layout in GenerateGlobalTensorTypeDeclaration
  std::set<std::string> dn_tensors_;  ///< Tensor names loaded with layout="dn" (need Layout::DN)
  std::map<std::string, std::string> tile_addresses_;  ///< tile_name → TASSIGN address expression

  // Loop tile hoisting: declarations collected during loop body visit, emitted before outermost loop
  int loop_depth_ = 0;                        ///< Current for-loop nesting depth (0 = not in loop)
  int if_depth_ = 0;                          ///< Current if-stmt nesting depth (0 = not in if)
  bool in_vf_scope_ = false;                  ///< Whether currently inside a __VEC_SCOPE__ block
  std::set<std::string> vf_ptr_vars_;         ///< Variable names that are __ubuf__ pointers in VF scope
  std::map<std::string, std::string> vf_post_update_ptrs_;  ///< tile_name → declared POST_UPDATE pointer var
  std::vector<std::string> loop_hoisted_decls_;  ///< Lines to hoist before outermost loop/if
  std::vector<std::string> cross_section_decls_;  ///< Decls needed by both Cube and Vec sections
  std::set<std::string> buf_id_var_names_;  ///< Var names used as runtime buf_id (Mutex)
  std::set<std::string> var_read_names_;    ///< Var names read anywhere in the function body
  std::unordered_map<std::string, int> var_read_counts_;  ///< Var name → read count
  std::map<int, std::set<ir::PipeType>> cube_mutex_pipes_;  ///< buf_id → pipes in Cube section
  std::map<int, std::set<ir::PipeType>> vec_mutex_pipes_;   ///< buf_id → pipes in Vec section

  /**
   * @brief Pre-scan the IR for Var names used as runtime buf_id (Mutex).
   *
   * A runtime buf_id is the first argument of ``system.mutex_lock_dyn`` or
   * ``system.mutex_unlock_dyn``. The N-way optimizer uses the result to emit
   * ``uint8_t``/``_bid_`` arrays for these Vars instead of ``event_t``/``_eid_``
   * arrays, avoiding semantic confusion with event_id arrays.
   */
  void CollectBufIdVarNames(const ir::StmtPtr& stmt, std::set<std::string>& out) const;

  /**
   * @brief Pre-scan the IR for mutex_id → pipe mappings.
   *
   * Used on A5 to eliminate redundant PIPE_V get_buf/rls_buf: if a buf_id is
   * only used by PIPE_V (never by MTE2/MTE3/M/etc.), the V-side mutex
   * synchronization is unnecessary (V→V ops execute in order within the pipe).
   */
  void CollectMutexPipeInfo(const ir::StmtPtr& stmt);

  /**
   * @brief Pre-scan the IR for all Var names appearing in read positions
   * (Call args, Yield values, AssignStmt/ReturnStmt values, subscripts, etc.).
   *
   * Used to drop IfStmt phi return_vars that have no downstream consumer —
   * the SSA pass conservatively inserts phi nodes whenever a variable is
   * re-assigned across control flow, but if the resulting phi var is never
   * read afterwards, EmitFullPhiIf would emit a dead declaration and
   * unused branch assignments.
   */
  void CollectVarReadNames(const ir::StmtPtr& stmt, std::set<std::string>& out) const;

  // EventId array deduplication: maps (val0, val1) → EventId variable name (2-way)
  std::map<std::pair<int64_t, int64_t>, std::string> event_id_decls_;
  // EventId N-way array deduplication: maps "index_expr:v0,v1,v2,..." → EventId variable name
  std::map<std::string, std::string> event_id_decls_nway_;
  std::set<std::string> event_id_names_used_;  // track used names to avoid redefinition
  int event_id_nway_counter_ = 0;

  // Tile array deduplication: maps "tile0,tile1" → shared Tile array name
  std::map<std::string, std::string> tile_array_decls_;
  int tile_array_counter_ = 0;  ///< Counter for unique Tile array names
  int tile_offset_counter_ = 0;  ///< Counter for unique TileOffsetExpr temp names

  // Tile type dedup: maps tile_type_str → alias name already emitted
  std::map<std::string, std::string> emitted_tile_types_;

  bool section_snapshot_saved_ = false;  ///< Whether context snapshot has been saved before first section

  /// Struct type dedup: maps field signature (CSV) → type name.
  /// Identical structs share the same type definition.
  std::map<std::string, std::string> struct_type_defs_;

  /// Track emitted tile reference aliases to avoid C++ redefinition errors
  std::set<std::string> emitted_tile_aliases_;

  // BufferSlot optimization: merge NBuffer tile+bid arrays into struct arrays
  struct PendingTileNWay {
    std::string index_expr;
    std::string arr_name;
    std::string tile_type_str;
    std::vector<std::string> tile_vals;
    ir::VarPtr return_var;
    std::string dedup_key;
  };
  std::optional<PendingTileNWay> pending_tile_nway_;
  bool buffer_slot_struct_emitted_ = false;
  std::set<std::string> buffer_slot_decls_emitted_;  ///< dedup for BufferSlot array decls

  void FlushPendingTileNWay();
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_
