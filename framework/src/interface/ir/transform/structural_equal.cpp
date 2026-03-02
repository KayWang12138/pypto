/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/reflection/field_visitor.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/printer.h"
#include "ir/transform/structural_comparison.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Unified structural equality checker for IR nodes
 *
 * Template parameter controls behavior on mismatch:
 * - AssertMode=false: Returns false (for structural_equal)
 * - AssertMode=true: Throws ValueError with detailed error message (for assert_structural_equal)
 *
 * This class is not part of the public API - use structural_equal() or assert_structural_equal().
 *
 * Implements the FieldIterator visitor interface for generic field-based comparison.
 * Uses the dual-node Visit overload which calls visitor methods with two field arguments.
 */
template <bool AssertMode>
class StructuralEqualImpl {
 public:
  using resultType = bool;

  explicit StructuralEqualImpl(bool enableAutoMapping) : enableAutoMapping_(enableAutoMapping) {}

  // Returns bool for structural_equal, throws for assert_structural_equal
  bool operator()(const IRNodePtr &lhs, const IRNodePtr &rhs) {
    if constexpr (AssertMode) {
      Equal(lhs, rhs);
      return true;  // Only reached if no exception thrown
    } else {
      return Equal(lhs, rhs);
    }
  }

  bool operator()(const TypePtr &lhs, const TypePtr &rhs) {
    if constexpr (AssertMode) {
      EqualType(lhs, rhs);
      return true;  // Only reached if no exception thrown
    } else {
      return EqualType(lhs, rhs);
    }
  }

  // FieldIterator visitor interface (dual-node version - methods receive two fields)
  [[nodiscard]] resultType InitResult() const { return true; }

  template <typename IRNodePtrType>
  resultType VisitIRNodeField(const IRNodePtrType &lhs, const IRNodePtrType &rhs) {
    INTERNAL_CHECK(lhs) << "structural_equal encountered null lhs IR node field";
    INTERNAL_CHECK(rhs) << "structural_equal encountered null rhs IR node field";
    return Equal(lhs, rhs);
  }

  // Specialization for std::optional<IRNodePtr>
  template <typename IRNodePtrType>
  resultType VisitIRNodeField(const std::optional<IRNodePtrType>& lhs,
                               const std::optional<IRNodePtrType>& rhs) {
    if (!lhs.has_value() && !rhs.has_value()) {
      return true;
    }
    if (!lhs.has_value() || !rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Optional field presence mismatch", lhs.has_value() ? *lhs : IRNodePtr(),
                      rhs.has_value() ? *rhs : IRNodePtr(), lhs.has_value() ? "has value" : "nullopt",
                      rhs.has_value() ? "has value" : "nullopt");
      }
      return false;
    }
    if (!*lhs && !*rhs) {
      return true;
    }
    if (!*lhs || !*rhs) {
      if constexpr (AssertMode) {
        ThrowMismatch("Optional field nullptr mismatch", *lhs, *rhs, *lhs ? "has value" : "nullptr",
                      *rhs ? "has value" : "nullptr");
      }
      return false;
    }
    return Equal(*lhs, *rhs);
  }

  template <typename IRNodePtrType>
  resultType VisitIRNodeVectorField(const std::vector<IRNodePtrType>& lhs,
                                     const std::vector<IRNodePtrType>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Vector size mismatch (" << lhs.size() << " items != " << rhs.size() << " items)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs IR node in vector at index " << i;
      INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs IR node in vector at index " << i;

      if constexpr (AssertMode) {
        std::ostringstream indexStr;
        indexStr << "[" << i << "]";
        path_.push_back(indexStr.str());
      }

      if (!Equal(lhs[i], rhs[i])) {
        if constexpr (AssertMode) {
          path_.pop_back();
        }
        return false;
      }

      if constexpr (AssertMode) {
        path_.pop_back();
      }
    }
    return true;
  }

  template <typename KeyType, typename ValueType, typename Compare>
  resultType VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& lhs,
                                  const std::map<KeyType, ValueType, Compare>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Map size mismatch (" << lhs.size() << " items != " << rhs.size() << " items)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    auto lhsIt = lhs.begin();
    auto rhsIt = rhs.begin();
    while (lhsIt != lhs.end()) {
      INTERNAL_CHECK(lhsIt->first) << "structural_equal encountered null lhs key in map";
      INTERNAL_CHECK(lhsIt->second) << "structural_equal encountered null lhs value in map";
      INTERNAL_CHECK(rhsIt->first) << "structural_equal encountered null rhs key in map";
      INTERNAL_CHECK(rhsIt->second) << "structural_equal encountered null rhs value in map";

      if (lhsIt->first->name_ != rhsIt->first->name_) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Map key mismatch ('" << lhsIt->first->name_ << "' != '" << rhsIt->first->name_ << "')";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }

      if constexpr (AssertMode) {
        std::ostringstream keyStr;
        keyStr << "['" << lhsIt->first->name_ << "']";
        path_.push_back(keyStr.str());
      }

      if (!Equal(lhsIt->second, rhsIt->second)) {
        if constexpr (AssertMode) {
          path_.pop_back();
        }
        return false;
      }

      if constexpr (AssertMode) {
        path_.pop_back();
      }
      ++lhsIt;
      ++rhsIt;
    }
    return true;
  }

  // Leaf field comparisons (dual-node version)
  resultType VisitLeafField(const int &lhs, const int &rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Integer value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  resultType VisitLeafField(const int64_t &lhs, const int64_t &rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "int64_t value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  resultType VisitLeafField(const uint64_t &lhs, const uint64_t &rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "uint64_t value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  resultType VisitLeafField(const double &lhs, const double &rhs) {
    if (std::memcmp(&lhs, &rhs, sizeof(double)) != 0) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "double value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  resultType VisitLeafField(const std::string &lhs, const std::string &rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "String value mismatch (\"" << lhs << "\" != \"" << rhs << "\")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  resultType VisitLeafField(const OpPtr &lhs, const OpPtr &rhs) {
    if (lhs->name_ != rhs->name_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Operator name mismatch ('" << lhs->name_ << "' != '" << rhs->name_ << "')";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  resultType VisitLeafField(const DataType &lhs, const DataType &rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "DataType mismatch (" << lhs.ToString() << " != " << rhs.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  resultType VisitLeafField(const FunctionType &lhs, const FunctionType &rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "FunctionType mismatch (" << FunctionTypeToString(lhs) << " != " << FunctionTypeToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  // Compare kwargs (vector of pairs to preserve order)
  resultType VisitLeafField(const std::vector<std::pair<std::string, std::any>>& lhs,
                             const std::vector<std::pair<std::string, std::any>>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Kwargs size mismatch (" << lhs.size() << " != " << rhs.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i].first != rhs[i].first) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs key mismatch at index " << i << " ('" << lhs[i].first << "' != '" << rhs[i].first
              << "')";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare std::any values by type and content
      const auto &lhsVal = lhs[i].second;
      const auto &rhsVal = rhs[i].second;
      if (lhsVal.type() != rhsVal.type()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs value type mismatch for key '" << lhs[i].first << "'";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Type-specific comparison
      bool valuesEqual = true;
      if (lhsVal.type() == typeid(int)) {
        valuesEqual = (AnyCast<int>(lhsVal, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<int>(rhsVal, "comparing kwarg: " + lhs[i].first));
      } else if (lhsVal.type() == typeid(bool)) {
        valuesEqual = (AnyCast<bool>(lhsVal, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<bool>(rhsVal, "comparing kwarg: " + lhs[i].first));
      } else if (lhsVal.type() == typeid(std::string)) {
        valuesEqual = (AnyCast<std::string>(lhsVal, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<std::string>(rhsVal, "comparing kwarg: " + lhs[i].first));
      } else if (lhsVal.type() == typeid(double)) {
        double lhsDouble = AnyCast<double>(lhsVal, "comparing kwarg: " + lhs[i].first);
        double rhsDouble = AnyCast<double>(rhsVal, "comparing kwarg: " + lhs[i].first);
        valuesEqual = (std::memcmp(&lhsDouble, &rhsDouble, sizeof(double)) == 0);
      } else if (lhsVal.type() == typeid(DataType)) {
        valuesEqual = (AnyCast<DataType>(lhsVal, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<DataType>(rhsVal, "comparing kwarg: " + lhs[i].first));
      }
      if (!valuesEqual) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs value mismatch for key '" << lhs[i].first << "'";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
    }
    return true;
  }

  resultType VisitLeafField(const MemorySpace &lhs, const MemorySpace &rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "MemorySpace mismatch (" << MemorySpaceToString(lhs) << " != " << MemorySpaceToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  resultType VisitLeafField(const TypePtr &lhs, const TypePtr &rhs) { return EqualType(lhs, rhs); }

  resultType VisitLeafField(const std::vector<TypePtr>& lhs, const std::vector<TypePtr>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Type vector size mismatch (" << lhs.size() << " types != " << rhs.size() << " types)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs TypePtr in vector at index " << i;
      INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs TypePtr in vector at index " << i;
      if (!EqualType(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  resultType VisitLeafField(const Span& /*lhs*/, const Span& /*rhs*/) const {
    INTERNAL_UNREACHABLE << "structural_equal should not visit Span field";
    return true;  // Never reached
  }

  // Field kind hooks
  template <typename FVisitOp>
  void VisitIgnoreField([[maybe_unused]] FVisitOp &&visitOp) {
    // Ignored fields are always considered equal
  }

  template <typename FVisitOp>
  void VisitDefField(FVisitOp &&visitOp) {
    bool enableAutoMapping = true;
    std::swap(enableAutoMapping, enableAutoMapping_);
    visitOp();
    std::swap(enableAutoMapping, enableAutoMapping_);
  }

  template <typename FVisitOp>
  void VisitUsualField(FVisitOp &&visitOp) {
    visitOp();
  }

  // Combine results (AND logic)
  template <typename Desc>
  void CombineResult(resultType &accumulator, resultType fieldResult, [[maybe_unused]] const Desc &desc) {
    accumulator = accumulator && fieldResult;
  }

 private:
  bool Equal(const IRNodePtr &lhs, const IRNodePtr &rhs);
  bool EqualVar(const VarPtr &lhs, const VarPtr &rhs);
  bool EqualIterArg(const IterArgPtr &lhs, const IterArgPtr &rhs);
  bool EqualType(const TypePtr &lhs, const TypePtr &rhs);

  /**
   * @brief Generic field-based equality check for IR nodes using FieldIterator
   *
   * Uses the dual-node Visit overload which passes two fields to each visitor method.
   *
   * @tparam NodePtr Shared pointer type to the node
   * @param lhs_op Left-hand side node
   * @param rhs_op Right-hand side node
   * @return true if all fields are equal
   */
  template <typename NodePtr>
  bool EqualWithFields(const NodePtr &lhsOp, const NodePtr &rhsOp) {
    using NodeType = typename NodePtr::element_type;
    auto descriptors = NodeType::GetFieldDescriptors();

    return std::apply(
        [&](auto&&... descs) {
          return reflection::FieldIterator<NodeType, StructuralEqualImpl<AssertMode>,
                                           decltype(descs)...>::Visit(*lhsOp, *rhsOp, *this, descs...);
        },
        descriptors);
  }

  // Only used in assert mode for error messages
  void ThrowMismatch(const std::string &reason, const IRNodePtr &lhs, const IRNodePtr &rhs,
                     const std::string &lhsDesc = "", const std::string &rhsDesc = "") {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Structural equality assertion failed";

      if (!path_.empty()) {
        msg << " at: ";
        for (size_t i = 0; i < path_.size(); ++i) {
          msg << path_[i];
          if (i < path_.size() - 1 && path_[i + 1][0] != '[') {
            msg << ".";
          }
        }
      }
      msg << "\n\n";

      if (lhs || rhs) {
        msg << "Left-hand side:\n";
        if (lhs) {
          std::string lhsStr = PythonPrint(lhs, "pl");
          std::istringstream iss(lhsStr);
          std::string line;
          while (std::getline(iss, line)) {
            msg << "  " << line << "\n";
          }
        } else {
          msg << "  (null)\n";
        }

        msg << "\nRight-hand side:\n";
        if (rhs) {
          std::string rhsStr = PythonPrint(rhs, "pl");
          std::istringstream iss(rhsStr);
          std::string line;
          while (std::getline(iss, line)) {
            msg << "  " << line << "\n";
          }
        } else {
          msg << "  (null)\n";
        }
        msg << "\n";
      } else if (!lhsDesc.empty() || !rhsDesc.empty()) {
        msg << "Left: " << lhsDesc << "\n";
        msg << "Right: " << rhsDesc << "\n\n";
      }

      msg << "Reason: " << reason;
      throw ValueError(msg.str());
    }
  }

  bool enableAutoMapping_;
  std::unordered_map<VarPtr, VarPtr> lhsToRhsVarMap_;
  std::unordered_map<VarPtr, VarPtr> rhsToLhsVarMap_;
  std::vector<std::string> path_;  // Only used in assert mode
};

// Type dispatch macro for generic field-based comparison
#define EQUAL_DISPATCH(Type)                                             \
  if (auto lhs_##Type = As<Type>(lhs)) {                                 \
    if constexpr (AssertMode) path_.emplace_back(#Type);                 \
    auto rhs_##Type = As<Type>(rhs);                                     \
    bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
    if constexpr (AssertMode) path_.pop_back();                          \
    return result;                                                       \
  }

// Dispatch macro for abstract base classes
#define EQUAL_DISPATCH_BASE(Type)                                        \
  if (auto lhs_##Type = As<Type>(lhs)) {                                 \
    if constexpr (AssertMode) path_.emplace_back(#Type);                 \
    auto rhs_##Type = As<Type>(rhs);                                     \
    bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
    if constexpr (AssertMode) path_.pop_back();                          \
    return result;                                                       \
  }

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::Equal(const IRNodePtr &lhs, const IRNodePtr &rhs) {
  if (lhs.get() == rhs.get()) return true;

  if (!lhs || !rhs) {
    if constexpr (AssertMode) ThrowMismatch("One node is null, the other is not", lhs, rhs);
    return false;
  }

  if (lhs->TypeName() != rhs->TypeName()) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Node type mismatch (" << lhs->TypeName() << " != " << rhs->TypeName() << ")";
      ThrowMismatch(msg.str(), lhs, rhs);
    }
    return false;
  }

  // Check IterArg before Var (IterArg inherits from Var)
  if (auto lhsIter = As<IterArg>(lhs)) {
    if constexpr (AssertMode) path_.emplace_back("IterArg");
    bool result = EqualIterArg(lhsIter, std::static_pointer_cast<const IterArg>(rhs));
    if constexpr (AssertMode) path_.pop_back();
    return result;
  }

  // Check MemRef before Var (MemRef inherits from Var)
  if (auto lhsMemref = As<MemRef>(lhs)) {
    if constexpr (AssertMode) path_.emplace_back("MemRef");
    auto rhsMemref = std::static_pointer_cast<const MemRef>(rhs);
    bool result = rhsMemref && EqualWithFields(lhsMemref, rhsMemref);
    if constexpr (AssertMode) path_.pop_back();
    return result;
  }

  if (auto lhsVar = As<Var>(lhs)) {
    if constexpr (AssertMode) path_.emplace_back("Var");
    bool result = EqualVar(lhsVar, std::static_pointer_cast<const Var>(rhs));
    if constexpr (AssertMode) path_.pop_back();
    return result;
  }

  // All other types use generic field-based comparison
  EQUAL_DISPATCH(ConstInt)
  EQUAL_DISPATCH(ConstFloat)
  EQUAL_DISPATCH(ConstBool)
  EQUAL_DISPATCH(Call)
  EQUAL_DISPATCH(MakeTuple)
  EQUAL_DISPATCH(TupleGetItemExpr)

  // BinaryExpr and UnaryExpr are abstract base classes, use dynamic_pointer_cast
  EQUAL_DISPATCH_BASE(BinaryExpr)
  EQUAL_DISPATCH_BASE(UnaryExpr)

  EQUAL_DISPATCH(AssignStmt)
  EQUAL_DISPATCH(IfStmt)
  EQUAL_DISPATCH(YieldStmt)
  EQUAL_DISPATCH(ReturnStmt)
  EQUAL_DISPATCH(ForStmt)
  EQUAL_DISPATCH(SeqStmts)
  EQUAL_DISPATCH(OpStmts)
  EQUAL_DISPATCH(EvalStmt)
  EQUAL_DISPATCH(Function)
  EQUAL_DISPATCH(Program)

  throw TypeError("Unknown IR node type in StructuralEqualImpl::Equal: " + lhs->TypeName());
}

#undef EQUAL_DISPATCH
#undef EQUAL_DISPATCH_BASE

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualType(const TypePtr &lhs, const TypePtr &rhs) {
  if (lhs->TypeName() != rhs->TypeName()) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Type name mismatch (" << lhs->TypeName() << " != " << rhs->TypeName() << ")";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  if (auto lhsScalar = As<ScalarType>(lhs)) {
    auto rhsScalar = As<ScalarType>(rhs);
    if (!rhsScalar) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for ScalarType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhsScalar->dtype_ != rhsScalar->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ScalarType dtype mismatch (" << lhsScalar->dtype_.ToString()
            << " != " << rhsScalar->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  } else if (auto lhsTensor = As<TensorType>(lhs)) {
    auto rhsTensor = As<TensorType>(rhs);
    if (!rhsTensor) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TensorType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhsTensor->dtype_ != rhsTensor->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TensorType dtype mismatch (" << lhsTensor->dtype_.ToString()
            << " != " << rhsTensor->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhsTensor->shape_.size() != rhsTensor->shape_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TensorType shape rank mismatch (" << lhsTensor->shape_.size()
            << " != " << rhsTensor->shape_.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhsTensor->shape_.size(); ++i) {
      if (!Equal(lhsTensor->shape_[i], rhsTensor->shape_[i])) return false;
    }
    return true;
  } else if (auto lhsTile = As<TileType>(lhs)) {
    auto rhsTile = As<TileType>(rhs);
    if (!rhsTile) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TileType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare dtype
    if (lhsTile->dtype_ != rhsTile->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TileType dtype mismatch (" << lhsTile->dtype_.ToString()
            << " != " << rhsTile->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare shape size and dimensions
    if (lhsTile->shape_.size() != rhsTile->shape_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TileType shape rank mismatch (" << lhsTile->shape_.size()
            << " != " << rhsTile->shape_.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhsTile->shape_.size(); ++i) {
      if (!Equal(lhsTile->shape_[i], rhsTile->shape_[i])) return false;
    }
    // Compare tile_view
    if (lhsTile->tileView_.has_value() != rhsTile->tileView_.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileType tile_view presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhsTile->tileView_.has_value()) {
      const auto &lhsTv = lhsTile->tileView_.value();
      const auto &rhsTv = rhsTile->tileView_.value();
      // Compare valid_shape
      if (lhsTv.validShape.size() != rhsTv.validShape.size()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "TileView valid_shape size mismatch (" << lhsTv.validShape.size()
              << " != " << rhsTv.validShape.size() << ")";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      for (size_t i = 0; i < lhsTv.validShape.size(); ++i) {
        if (!Equal(lhsTv.validShape[i], rhsTv.validShape[i])) return false;
      }
      // Compare stride
      if (lhsTv.stride.size() != rhsTv.stride.size()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "TileView stride size mismatch (" << lhsTv.stride.size() << " != " << rhsTv.stride.size()
              << ")";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      for (size_t i = 0; i < lhsTv.stride.size(); ++i) {
        if (!Equal(lhsTv.stride[i], rhsTv.stride[i])) return false;
      }
      // Compare start_offset
      if (!Equal(lhsTv.startOffset, rhsTv.startOffset)) return false;
    }
    return true;
  } else if (auto lhsTuple = As<TupleType>(lhs)) {
    auto rhsTuple = As<TupleType>(rhs);
    if (!rhsTuple) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TupleType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhsTuple->types_.size() != rhsTuple->types_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TupleType size mismatch (" << lhsTuple->types_.size() << " != " << rhsTuple->types_.size()
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhsTuple->types_.size(); ++i) {
      if (!EqualType(lhsTuple->types_[i], rhsTuple->types_[i])) return false;
    }
    return true;
  } else if (IsA<MemRefType>(lhs)) {
    // MemRefType is a singleton type, just need to check both are MemRefType
    return true;
  } else if (IsA<UnknownType>(lhs)) {
    return true;
  }

  INTERNAL_UNREACHABLE << "EqualType encountered unhandled Type: " << lhs->TypeName();
  return false;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualVar(const VarPtr &lhs, const VarPtr &rhs) {
  if (!enableAutoMapping_) {
    auto lhsIt = lhsToRhsVarMap_.find(lhs);
    auto rhsIt = rhsToLhsVarMap_.find(rhs);
    // Case 1: already mapped to the same variable
    if (lhsIt != lhsToRhsVarMap_.end() && rhsIt != rhsToLhsVarMap_.end()) {
      if (lhsIt->second != rhs || rhsIt->second != lhs) {
        if constexpr (AssertMode) {
          ThrowMismatch("Variable mapping inconsistent (without auto-mapping)",
                        std::static_pointer_cast<const IRNode>(lhs),
                        std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_,
                        "var " + rhs->name_);
        }
        return false;
      }
      return true;
    }
    // Case 2: different variables
    if (lhs.get() != rhs.get()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Variable pointer mismatch (without auto-mapping)",
                      std::static_pointer_cast<const IRNode>(lhs),
                      std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_, "var " + rhs->name_);
      }
      return false;
    }
    return true;
  }

  if (!EqualType(lhs->GetType(), rhs->GetType())) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Variable type mismatch (" << lhs->GetType()->TypeName() << " != " << rhs->GetType()->TypeName()
          << ")";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  auto it = lhsToRhsVarMap_.find(lhs);
  if (it != lhsToRhsVarMap_.end()) {
    if (it->second != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Variable mapping inconsistent ('" << lhs->name_ << "' cannot map to both '"
            << it->second->name_ << "' and '" << rhs->name_ << "')";
        ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                      std::static_pointer_cast<const IRNode>(rhs));
      }
      return false;
    }
    return true;
  }

  auto rhsIt = rhsToLhsVarMap_.find(rhs);
  if (rhsIt != rhsToLhsVarMap_.end() && rhsIt->second != lhs) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Variable mapping inconsistent ('" << rhs->name_ << "' is already mapped from '"
          << rhsIt->second->name_ << "', cannot map from '" << lhs->name_ << "')";
      ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  lhsToRhsVarMap_[lhs] = rhs;
  rhsToLhsVarMap_[rhs] = lhs;
  return true;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualIterArg(const IterArgPtr &lhs, const IterArgPtr &rhs) {
  // 1. First, compare as Var (handles variable mapping)
  if (!EqualVar(lhs, rhs)) {
    return false;
  }

  // 2. Then, compare IterArg-specific field: initValue_
  if (!Equal(lhs->initValue_, rhs->initValue_)) {
    if constexpr (AssertMode) {
      ThrowMismatch("IterArg initValue mismatch", std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  return true;
}

// Explicit template instantiations
template class StructuralEqualImpl<false>;  // For structural_equal
template class StructuralEqualImpl<true>;   // For assert_structural_equal

// Type aliases for cleaner code
using StructuralEqualChecker = StructuralEqualImpl<false>;
using StructuralEqualAssert = StructuralEqualImpl<true>;

// Public API implementation
bool StructuralEqual(const IRNodePtr &lhs, const IRNodePtr &rhs, bool enableAutoMapping) {
  StructuralEqualChecker checker(enableAutoMapping);
  return checker(lhs, rhs);
}

bool StructuralEqual(const TypePtr &lhs, const TypePtr &rhs, bool enableAutoMapping) {
  StructuralEqualChecker checker(enableAutoMapping);
  return checker(lhs, rhs);
}

// Public assert API
void AssertStructuralEqual(const IRNodePtr &lhs, const IRNodePtr &rhs, bool enableAutoMapping) {
  StructuralEqualAssert checker(enableAutoMapping);
  checker(lhs, rhs);
}

void AssertStructuralEqual(const TypePtr &lhs, const TypePtr &rhs, bool enableAutoMapping) {
  StructuralEqualAssert checker(enableAutoMapping);
  checker(lhs, rhs);
}

}  // namespace ir
}  // namespace pypto
