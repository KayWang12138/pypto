/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file functor.h
 * \brief Functor classes for visiting IR nodes
 */

#pragma once

#include "ir/statement.h"
#include "ir/value.h"
#include "ir/operation_base.h"
#include "ir/function.h"
#include "ir/program.h"

#include <memory>
#include <stdexcept>

namespace pto {

// Forward declarations
class ProgramModule;
class Function;
class Statement;
class Operation;
class Value;

// Type aliases for functor parameters
using ProgramModulePtr = std::shared_ptr<ProgramModule>;
using FunctionPtr = std::shared_ptr<Function>;
using TensorValuePtr = std::shared_ptr<TensorValue>;

/**
 * @brief Base template for program functors
 *
 * Provides a visitor-like interface for operating on IR programs.
 * Subclasses implement specific operations by overriding VisitProgram_ methods.
 *
 * @tparam R Return type of the visit operations
 * @tparam Args Additional arguments passed to visit methods
 */
template <typename R, typename... Args>
class ProgramFunctor {
 public:
  virtual ~ProgramFunctor() = default;

  /**
   * @brief Dispatcher for program types
   *
   * Uses dynamic_cast to determine concrete type and dispatch to appropriate handler.
   *
   * @param program Program pointer (non-null)
   * @param args Additional arguments
   * @return Result of visiting the program
   */
  virtual R VisitProgram(ProgramModulePtr& program, Args... args);

 protected:
  // Program types
  virtual R VisitProgram_(ProgramModulePtr& program, Args... args) = 0;
};

// Macro to dispatch based on program type
#define PROGRAM_FUNCTOR_DISPATCH(OpType)                            \
  if (auto op = std::dynamic_pointer_cast<OpType>(program)) { \
    return VisitProgram_(op, std::forward<Args>(args)...);          \
  }

template <typename R, typename... Args>
R ProgramFunctor<R, Args...>::VisitProgram(ProgramModulePtr& program, Args... args) {
  // Dispatch to concrete program types
  PROGRAM_FUNCTOR_DISPATCH(ProgramModule);

  // Should never reach here if all types are handled
  throw std::runtime_error("Unknown program type in ProgramFunctor::VisitProgram");
}

#undef PROGRAM_FUNCTOR_DISPATCH

/**
 * @brief Base template for function functors
 *
 * Provides a visitor-like interface for operating on IR functions.
 * Subclasses implement specific operations by overriding VisitFunction_ methods.
 *
 * @tparam R Return type of the visit operations
 * @tparam Args Additional arguments passed to visit methods
 */
template <typename R, typename... Args>
class FunctionFunctor {
 public:
  virtual ~FunctionFunctor() = default;

  /**
   * @brief Dispatcher for function types
   *
   * Uses dynamic_cast to determine concrete type and dispatch to appropriate handler.
   *
   * @param func Function pointer (non-null)
   * @param args Additional arguments
   * @return Result of visiting the function
   */
  virtual R VisitFunction(FunctionPtr& func, Args... args);

 protected:
  // Function types
  virtual R VisitFunction_(FunctionPtr& func, Args... args) = 0;
};

// Macro to dispatch based on function type
#define FUNCTION_FUNCTOR_DISPATCH(OpType)                            \
  if (auto op = std::dynamic_pointer_cast<OpType>(func)) { \
    return VisitFunction_(op, std::forward<Args>(args)...);          \
  }

template <typename R, typename... Args>
R FunctionFunctor<R, Args...>::VisitFunction(FunctionPtr& func, Args... args) {
  // Dispatch to concrete function types
  FUNCTION_FUNCTOR_DISPATCH(Function);

  // Should never reach here if all types are handled
  throw std::runtime_error("Unknown function type in FunctionFunctor::VisitFunction");
}

#undef FUNCTION_FUNCTOR_DISPATCH

/**
 * @brief Base template for statement functors
 *
 * Provides a visitor-like interface for operating on IR statements.
 * Subclasses implement specific operations by overriding VisitStmt_ methods.
 *
 * @tparam R Return type of the visit operations
 * @tparam Args Additional arguments passed to visit methods
 */
template <typename R, typename... Args>
class StatementFunctor {
 public:
  virtual ~StatementFunctor() = default;

  /**
   * @brief Dispatcher for statement types
   *
   * Uses dynamic_cast to determine concrete type and dispatch to appropriate handler.
   *
   * @param stmt Statement pointer (non-null)
   * @param args Additional arguments
   * @return Result of visiting the statement
   */
  virtual R VisitStmt(StatementPtr& stmt, Args... args);

 protected:
  // Statement types
  virtual R VisitStmt_(CompoundStatementPtr& op, Args... args) = 0;
  virtual R VisitStmt_(OpStatementPtr& op, Args... args) = 0;
  virtual R VisitStmt_(ForStatementPtr& op, Args... args) = 0;
  virtual R VisitStmt_(IfStatementPtr& op, Args... args) = 0;
  virtual R VisitStmt_(YieldStatementPtr& op, Args... args) = 0;
  virtual R VisitStmt_(ReturnStatementPtr& op, Args... args) = 0;
  virtual R VisitStmt_(StatementPtr& op, Args... args) = 0;
};

// Macro to dispatch based on statement type
#define STMT_FUNCTOR_DISPATCH(OpType)                            \
  if (auto op = std::dynamic_pointer_cast<OpType>(stmt)) { \
    return VisitStmt_(op, std::forward<Args>(args)...);          \
  }

template <typename R, typename... Args>
R StatementFunctor<R, Args...>::VisitStmt(StatementPtr& stmt, Args... args) {
  // Dispatch to concrete statement types
  STMT_FUNCTOR_DISPATCH(CompoundStatement);
  STMT_FUNCTOR_DISPATCH(OpStatement);
  STMT_FUNCTOR_DISPATCH(ForStatement);
  STMT_FUNCTOR_DISPATCH(IfStatement);
  STMT_FUNCTOR_DISPATCH(YieldStatement);
  STMT_FUNCTOR_DISPATCH(ReturnStatement);
  STMT_FUNCTOR_DISPATCH(Statement);

  // Should never reach here if all types are handled
  throw std::runtime_error("Unknown statement type in StatementFunctor::VisitStmt");
}

#undef STMT_FUNCTOR_DISPATCH

/**
 * @brief Base template for operation functors
 *
 * Provides a visitor-like interface for operating on IR operations.
 * Subclasses implement specific operations by overriding VisitOp_ methods.
 *
 * @tparam R Return type of the visit operations
 * @tparam Args Additional arguments passed to visit methods
 */
template <typename R, typename... Args>
class OperationFunctor {
 public:
  virtual ~OperationFunctor() = default;

  /**
   * @brief Dispatcher for operation types
   *
   * Uses dynamic_cast to determine concrete type and dispatch to appropriate handler.
   *
   * @param op Operation pointer (non-null)
   * @param args Additional arguments
   * @return Result of visiting the operation
   */
  virtual R VisitOp(OperationPtr& op, Args... args);

 protected:
  // Operation types
  virtual R VisitOp_(ScalarBaseOpPtr& op, Args... args) = 0;
  virtual R VisitOp_(OperationPtr& op, Args... args) = 0;
};

// Macro to dispatch based on operation type
#define OP_FUNCTOR_DISPATCH(OpType)                            \
  if (auto op_ptr = std::dynamic_pointer_cast<OpType>(op)) { \
    return VisitOp_(op_ptr, std::forward<Args>(args)...);          \
  }

template <typename R, typename... Args>
R OperationFunctor<R, Args...>::VisitOp(OperationPtr& op, Args... args) {
  // Dispatch to concrete operation types
  OP_FUNCTOR_DISPATCH(ScalarBaseOp);
  OP_FUNCTOR_DISPATCH(Operation);

  // Should never reach here if all types are handled
  throw std::runtime_error("Unknown operation type in OperationFunctor::VisitOp");
}

#undef OP_FUNCTOR_DISPATCH

/**
 * @brief Base template for value functors
 *
 * Provides a visitor-like interface for operating on IR values.
 * Subclasses implement specific operations by overriding VisitValue_ methods.
 *
 * @tparam R Return type of the visit operations
 * @tparam Args Additional arguments passed to visit methods
 */
template <typename R, typename... Args>
class ValueFunctor {
 public:
  virtual ~ValueFunctor() = default;

  /**
   * @brief Dispatcher for value types
   *
   * Uses dynamic_cast to determine concrete type and dispatch to appropriate handler.
   *
   * @param value Value pointer (non-null)
   * @param args Additional arguments
   * @return Result of visiting the value
   */
  virtual R VisitValue(ValuePtr& value, Args... args);

 protected:
  // Value types
  virtual R VisitValue_(ScalarValuePtr& op, Args... args) = 0;
  virtual R VisitValue_(TileValuePtr& op, Args... args) = 0;
  virtual R VisitValue_(TensorValuePtr& op, Args... args) = 0;
  virtual R VisitValue_(ValuePtr& op, Args... args) = 0;
};

// Macro to dispatch based on value type
#define VALUE_FUNCTOR_DISPATCH(OpType)                            \
  if (auto op = std::dynamic_pointer_cast<OpType>(value)) { \
    return VisitValue_(op, std::forward<Args>(args)...);          \
  }

template <typename R, typename... Args>
R ValueFunctor<R, Args...>::VisitValue(ValuePtr& value, Args... args) {
  // Dispatch to concrete value types
  VALUE_FUNCTOR_DISPATCH(ScalarValue);
  VALUE_FUNCTOR_DISPATCH(TileValue);
  VALUE_FUNCTOR_DISPATCH(TensorValue);
  VALUE_FUNCTOR_DISPATCH(Value);

  // Should never reach here if all types are handled
  throw std::runtime_error("Unknown value type in ValueFunctor::VisitValue");
}

#undef VALUE_FUNCTOR_DISPATCH

/**
 * @brief Aggregated functor for all IR node types
 *
 * Combines ProgramFunctor, FunctionFunctor, StatementFunctor, OperationFunctor,
 * and ValueFunctor into a single interface for visiting all IR node types.
 * Each functor can have its own return type.
 *
 * @tparam RProgram Return type for ProgramFunctor visit operations
 * @tparam RFunction Return type for FunctionFunctor visit operations
 * @tparam RStatement Return type for StatementFunctor visit operations
 * @tparam ROperation Return type for OperationFunctor visit operations
 * @tparam RValue Return type for ValueFunctor visit operations
 * @tparam Args Additional arguments passed to visit methods
 */
template <typename RProgram, typename RFunction, typename RStatement,
          typename ROperation, typename RValue, typename... Args>
class IRFunctor : public ProgramFunctor<RProgram, Args...>,
                  public FunctionFunctor<RFunction, Args...>,
                  public StatementFunctor<RStatement, Args...>,
                  public OperationFunctor<ROperation, Args...>,
                  public ValueFunctor<RValue, Args...> {
 public:
  virtual ~IRFunctor() = default;
};

} // namespace pto

