/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_IR_FUNCTION_H_
#define PYPTO_IR_FUNCTION_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ir/core.h"
#include "ir/expr.h"
#include "ir/reflection/field_traits.h"
#include "ir/stmt.h"
#include "ir/type.h"

// Forward declaration from interface layer
namespace npu::tile_fwk {
struct LeafFuncAttribute;
}

namespace pypto {
namespace ir {

/**
 * @brief Function definition (Temporary stub implementation)
 *
 * This is a minimal placeholder to maintain compilation compatibility with
 * structural_hash, structural_equal, and printer implementations.
 * Will be replaced with actual implementation from new IR.
 *
 * Represents a complete function definition with name, parameters, return types, and body.
 */
class Function : public IRNode {
 public:
  /**
   * @brief Default constructor - creates an empty function
   */
  Function() : IRNode(Span("", -1, -1)) {}

  /**
   * @brief Create a function definition
   *
   * @param name Function name
   * @param params Parameter variables
   * @param return_types Return types
   * @param body Function body statement (use SeqStmts for multiple statements)
   * @param span Source location
   */
  Function(std::string name, std::vector<VarPtr> params, std::vector<TypePtr> return_types, StmtPtr body,
           Span span = Span("", -1, -1))
      : IRNode(std::move(span)),
        name_(std::move(name)),
        params_(std::move(params)),
        return_types_(std::move(return_types)),
        body_(std::move(body)) {}

  [[nodiscard]] std::string TypeName() const override { return "Function"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * This enables structural_hash, structural_equal, and printer to work correctly.
   *
   * @return Tuple of field descriptors (params as DEF field, return_types and body as USUAL fields, name as
   * IGNORE field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(IRNode::GetFieldDescriptors(),
                          std::make_tuple(reflection::IgnoreField(&Function::name_, "name"),
                                          reflection::DefField(&Function::params_, "params"),
                                          reflection::UsualField(&Function::return_types_, "return_types"),
                                          reflection::UsualField(&Function::body_, "body")));
  }

 public:
  std::string name_;                   // Function name
  std::vector<VarPtr> params_;         // Parameter variables
  std::vector<TypePtr> return_types_;  // Return types
  StmtPtr body_;                       // Function body statement
};

using FunctionPtr = std::shared_ptr<const Function>;

}  // namespace ir
}  // namespace pypto

// Legacy namespace for backward compatibility
namespace pto {

/**
 * @brief Temporary stub for Function (Legacy)
 * This is a minimal placeholder to maintain compilation compatibility
 * Will be replaced with actual implementation from new IR
 */
class Function {
 public:
  Function() = default;
  explicit Function(std::string name) : name_(std::move(name)) {}
  virtual ~Function() = default;

  const std::string& GetName() const { return name_; }
  void SetName(const std::string& name) { name_ = name; }

  // Stub method for compatibility
  virtual std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> GetLeafFuncAttribute() const { return nullptr; }

 private:
  std::string name_;
};

/**
 * @brief Temporary stub for BlockFunction (Legacy)
 * This is a minimal placeholder to maintain compilation compatibility
 * Will be replaced with actual implementation from new IR
 */
class BlockFunction : public Function {
 public:
  BlockFunction() = default;
  explicit BlockFunction(std::string name) : Function(std::move(name)) {}

  // Stub method for compatibility
  std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> GetLeafFuncAttribute() const override { return leafAttr_; }

  void SetLeafFuncAttribute(std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> attr) { leafAttr_ = attr; }

  // Stub method for program ID - returns 0 as default
  int GetID() const { return id_; }
  void SetID(int id) { id_ = id; }

 private:
  std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> leafAttr_;
  int id_ = 0;
};

}  // namespace pto

#endif  // PYPTO_IR_FUNCTION_H_
