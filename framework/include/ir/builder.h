/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// Forward declarations
class BuildContext;
class FunctionContext;
class ForLoopBuilder;
class IfStmtContext;

/**
 * \brief IR Builder for incremental IR construction with context management
 *
 * The IRBuilder provides an API for building IR incrementally. It maintains a
 * context stack to track nested function and if-statement scopes. For loops are
 * built statelessly via CreateForLoop() returning a ForLoopBuilder.
 *
 * Key features:
 * - Stack-based context management for functions, if statements, and programs
 * - Stateless for loop building via CreateForLoop() and ForLoopBuilder
 * - All methods accept explicit Span parameters
 * - Validates proper nesting and construction
 *
 * Example usage (C++):
 * \code
 * IRBuilder builder;
 * auto span = Span(__FILE__, __LINE__, 0);
 * builder.BeginFunction("my_func", span);
 * auto x = builder.FuncArg("x", ScalarType::Create(DataType::INT64), span);
 * builder.ReturnType(ScalarType::Create(DataType::INT64));
 * // ... build body ...
 * auto func = builder.EndFunction(span);
 * \endcode
 */
class IRBuilder {
public:
    IRBuilder();
    ~IRBuilder();

    // Disable copying and moving since we have unique_ptr members
    IRBuilder(const IRBuilder &) = delete;
    IRBuilder &operator=(const IRBuilder &) = delete;
    IRBuilder(IRBuilder &&) = delete;
    IRBuilder &operator=(IRBuilder &&) = delete;

    // ========== Function Building ==========

    /**
     * \brief Begin building a function
     *
     * Creates a new function context and pushes it onto the context stack.
     * Must be closed with EndFunction().
     *
     * \param name Function name
     * \param span Source location for function definition
     * \param type Function type (default: Opaque)
     * \throws RuntimeError if already inside a function (no nested functions allowed)
     */
    void BeginFunction(const std::string &name, const Span &span, FunctionType type = FunctionType::OPAQUE);

    /**
     * \brief Add a function parameter
     *
     * Must be called within a function context (after BeginFunction).
     *
     * \param name Parameter name
     * \param type Parameter type
     * \param span Source location for parameter
     * \return Variable representing the parameter
     * \throws RuntimeError if not inside a function context
     */
    VarPtr FuncArg(const std::string &name, const TypePtr &type, const Span &span);

    /**
     * \brief Add a return type to the current function
     *
     * Can be called multiple times to add multiple return types.
     *
     * \param type Return type
     * \throws RuntimeError if not inside a function context
     */
    void ReturnType(const TypePtr &type);

    /**
     * \brief End building a function
     *
     * Finalizes the function and pops the function context from the stack.
     *
     * \param endSpan Source location for end of function
     * \return The built function
     * \throws RuntimeError if not inside a function context
     */
    FunctionPtr EndFunction(const Span &endSpan);

    // ========== For Loop Building ==========

    /**
     * \brief Create a for loop builder
     *
     * Returns a ForLoopBuilder for constructing a for loop without modifying the
     * builder's internal context stack. Use ForLoopBuilder::AddIterArg(),
     * ForLoopBuilder::AddReturnVar(), and ForLoopBuilder::Emit() to populate the
     * loop, then call ForLoopBuilder::Build() to produce the ForStmt.
     * Emit the resulting ForStmt to the current context with IRBuilder::Emit().
     *
     * \param loopVar Loop variable
     * \param start Start value expression
     * \param stop Stop value expression
     * \param step Step value expression
     * \param span Source location for loop definition
     * \return ForLoopBuilder for constructing the loop body
     */
    ForLoopBuilder CreateForLoop(
        const VarPtr &loopVar, const ExprPtr &start, const ExprPtr &stop, const ExprPtr &step, const Span &span);

    // ========== If Statement Building ==========

    /**
     * \brief Begin building an if statement
     *
     * Creates a new if context and pushes it onto the context stack.
     * Must be closed with EndIf().
     *
     * \param condition Condition expression
     * \param span Source location for if statement
     * \throws RuntimeError if not inside a function or loop
     */
    void BeginIf(const ExprPtr &condition, const Span &span);

    /**
     * \brief Begin the else branch of the current if statement
     *
     * Must be called after building the then branch and before EndIf().
     *
     * \param span Source location for else keyword
     * \throws RuntimeError if not inside an if context
     * \throws RuntimeError if else branch already begun
     */
    void BeginElse(const Span &span);

    /**
     * \brief Add a return variable to the current if statement
     *
     * Return variables are used for SSA phi nodes when if has return values.
     *
     * \param var Return variable
     * \throws RuntimeError if not inside an if context
     */
    void AddIfReturnVar(const VarPtr &var);

    /**
     * \brief End building an if statement
     *
     * Finalizes the if statement and pops the context from the stack.
     *
     * \param endSpan Source location for end of if
     * \return The built if statement
     * \throws RuntimeError if not inside an if context
     */
    StmtPtr EndIf(const Span &endSpan);

    // ========== Statement Recording ==========

    /**
     * \brief Emit a statement in the current context
     *
     * Adds a statement to the current context's statement list.
     *
     * \param stmt Statement to emit
     * \throws RuntimeError if not inside a valid context for emitting statements
     */
    void Emit(const StmtPtr &stmt);

    /**
     * \brief Create an assignment statement and emit it
     *
     * Convenience method that creates an assignment and emits it.
     *
     * \param var Variable to assign to
     * \param value Expression value
     * \param span Source location for assignment
     * \return The created assignment statement
     * \throws RuntimeError if not inside a valid context
     */
    AssignStmtPtr Assign(const VarPtr &var, const ExprPtr &value, const Span &span);

    /**
     * \brief Create a variable (does not emit)
     *
     * Helper to create a variable. User must create assignment separately.
     *
     * \param name Variable name
     * \param type Variable type
     * \param span Source location
     * \return The created variable
     */
    VarPtr Var(const std::string &name, const TypePtr &type, const Span &span);

    /**
     * \brief Create a return statement and emit it
     *
     * Convenience method that creates a return statement and emits it.
     *
     * \param values List of expressions to return (can be empty)
     * \param span Source location for return statement
     * \return The created return statement
     * \throws RuntimeError if not inside a valid context
     */
    ReturnStmtPtr Return(const std::vector<ExprPtr> &values, const Span &span);

    /**
     * \brief Create a return statement without values and emit it
     *
     * Convenience method that creates an empty return statement and emits it.
     *
     * \param span Source location for return statement
     * \return The created return statement
     * \throws RuntimeError if not inside a valid context
     */
    ReturnStmtPtr Return(const Span &span);

    // ========== Context State Queries ==========

    /**
     * \brief Get the current context
     *
     * \return Pointer to current context, or nullptr if no context
     */
    BuildContext *CurrentContext();

    /**
     * \brief Check if currently inside a function
     *
     * \return true if inside a function context
     */
    [[nodiscard]] bool InFunction() const;

    /**
     * \brief Check if currently inside an if statement
     *
     * \return true if inside an if statement context
     */
    [[nodiscard]] bool InIf() const;

    // ========== Program Building ==========

    /**
     * \brief Begin building a program
     *
     * Creates a new program context and pushes it onto the context stack.
     * Must be closed with EndProgram().
     *
     * \param name Program name
     * \param span Source location for program definition
     * \throws RuntimeError if already inside another program
     */
    void BeginProgram(const std::string &name, const Span &span);

    /**
     * \brief Declare a function in the current program
     *
     * Creates a GlobalVar for the function that can be used in Call expressions
     * before the function is fully built. This enables cross-function calls.
     *
     * \param funcName Function name to declare
     * \return GlobalVar that can be used in Call expressions
     * \throws RuntimeError if not inside a program context
     */
    GlobalVarPtr DeclareFunction(const std::string &funcName);

    /**
     * \brief Get a GlobalVar for a declared function
     *
     * Retrieves a GlobalVar that was previously declared with DeclareFunction.
     *
     * \param funcName Function name
     * \return GlobalVar for the function
     * \throws RuntimeError if not inside a program context or function not declared
     */
    GlobalVarPtr GetGlobalVar(const std::string &funcName);

    /**
     * \brief Add a completed function to the current program
     *
     * The function must have been previously declared with DeclareFunction.
     *
     * \param func Completed function to add
     * \throws RuntimeError if not inside a program context
     */
    void AddFunction(const FunctionPtr &func);

    /**
     * \brief End building a program
     *
     * Finalizes the program and pops the program context from the stack.
     *
     * \param endSpan Source location for end of program
     * \return The built program
     * \throws RuntimeError if not inside a program context
     */
    ProgramPtr EndProgram(const Span &endSpan);

    /**
     * \brief Check if currently inside a program
     *
     * \return true if inside a program context
     */
    [[nodiscard]] bool InProgram() const;

    /**
     * \brief Get return types for a function by its GlobalVar
     *
     * Returns the return types for a function if it has been added to the program.
     * Returns empty vector if not inside a program or function not yet added.
     *
     * \param gvar GlobalVar for the function
     * \return Vector of return types
     */
    [[nodiscard]] std::vector<TypePtr> GetFunctionReturnTypes(const GlobalVarPtr &gvar) const;

private:
    std::vector<std::unique_ptr<BuildContext>> contextStack_;

    // Helper to get current context with type checking
    template <typename T>
    T *GetCurrentContextAs();

    // Helper to validate we're in the right context
    void ValidateInFunction(const std::string &operation);
    void ValidateInIf(const std::string &operation);
    void ValidateInProgram(const std::string &operation);
};

/**
 * \brief Base class for build contexts
 *
 * Each context type (function, loop, if) maintains state for building
 * that construct incrementally.
 */
class BuildContext {
public:
    enum class Type { FUNCTION, IF_STMT, PROGRAM };

    explicit BuildContext(Type type, Span span) : type_(type), beginSpan_(std::move(span)) {}
    virtual ~BuildContext() = default;

    [[nodiscard]] Type GetType() const { return type_; }
    [[nodiscard]] const Span &GetBeginSpan() const { return beginSpan_; }

    // Accumulate statements in this context
    virtual void AddStmt(const StmtPtr &stmt) = 0;
    [[nodiscard]] const std::vector<StmtPtr> &GetStmts() const { return stmts_; }

protected:
    Type type_;
    Span beginSpan_;
    std::vector<StmtPtr> stmts_;
};

/**
 * \brief Context for building a function
 */
class FunctionContext : public BuildContext {
public:
    FunctionContext(std::string name, Span span, FunctionType funcType = FunctionType::OPAQUE)
        : BuildContext(Type::FUNCTION, std::move(span)), name_(std::move(name)), funcType_(funcType) {}

    void AddParam(const VarPtr &param) { params_.push_back(param); }
    void AddReturnType(const TypePtr &type) { returnTypes_.push_back(type); }

    void AddStmt(const StmtPtr &stmt) override { stmts_.push_back(stmt); }
    [[nodiscard]] const std::string &GetName() const { return name_; }
    [[nodiscard]] const std::vector<VarPtr> &GetParams() const { return params_; }
    [[nodiscard]] const std::vector<TypePtr> &GetReturnTypes() const { return returnTypes_; }
    [[nodiscard]] FunctionType GetFuncType() const { return funcType_; }

private:
    std::string name_;
    FunctionType funcType_;
    std::vector<VarPtr> params_;
    std::vector<TypePtr> returnTypes_;
};

/**
 * \brief Builder for constructing a for loop
 *
 * Provides a stateless API for building a for loop. Create with
 * IRBuilder::CreateForLoop(), then use AddIterArg(), AddReturnVar(), and
 * Emit() to populate the loop body. Finally call Build() to produce a ForStmt
 * which can be emitted to the parent context via IRBuilder::Emit().
 *
 * Example usage (C++):
 * \code
 * auto forLoop = builder.CreateForLoop(i, start, stop, step, span);
 * forLoop.AddIterArg(iterArg);
 * forLoop.AddReturnVar(retVar);
 * forLoop.Emit(someStmt);
 * auto forStmt = forLoop.Build(endSpan);
 * builder.Emit(forStmt);
 * \endcode
 */
class ForLoopBuilder {
public:
    ForLoopBuilder(VarPtr loopVar, ExprPtr start, ExprPtr stop, ExprPtr step, Span span)
        : loopVar_(std::move(loopVar)),
          start_(std::move(start)),
          stop_(std::move(stop)),
          step_(std::move(step)),
          beginSpan_(std::move(span)) {}

    /**
     * \brief Add an iteration argument to this for loop
     *
     * Iteration arguments are loop-carried values (SSA-style).
     *
     * \param iterArg Iteration argument with initial value
     */
    void AddIterArg(const IterArgPtr &iterArg) { iterArgs_.push_back(iterArg); }

    /**
     * \brief Add a return variable to this for loop
     *
     * Return variables capture the final values of iteration arguments.
     * The number of return variables must match the number of iteration arguments.
     *
     * \param var Return variable
     */
    void AddReturnVar(const VarPtr &var) { returnVars_.push_back(var); }

    /**
     * \brief Emit a statement into the loop body
     *
     * \param stmt Statement to add to the loop body
     */
    void Emit(const StmtPtr &stmt) { stmts_.push_back(stmt); }

    /**
     * \brief Build the ForStmt from accumulated state
     *
     * Finalizes the for loop and returns the built ForStmt.
     *
     * \param endSpan Source location for end of loop
     * \return The built for statement
     * \throws RuntimeError if number of return variables doesn't match iteration arguments
     */
    StmtPtr Build(const Span &endSpan);

private:
    VarPtr loopVar_;
    ExprPtr start_;
    ExprPtr stop_;
    ExprPtr step_;
    Span beginSpan_;
    std::vector<IterArgPtr> iterArgs_;
    std::vector<VarPtr> returnVars_;
    std::vector<StmtPtr> stmts_;
};

/**
 * \brief Context for building an if statement
 */
class IfStmtContext : public BuildContext {
public:
    IfStmtContext(ExprPtr condition, Span span)
        : BuildContext(Type::IF_STMT, std::move(span)), condition_(std::move(condition)) {}

    void BeginElseBranch() {
        inElseBranch_ = true;
        elseStmts_.clear();
    }

    void AddReturnVar(const VarPtr &var) { returnVars_.push_back(var); }

    void AddStmt(const StmtPtr &stmt) override {
        if (inElseBranch_) {
            elseStmts_.push_back(stmt);
        } else {
            stmts_.push_back(stmt);
        }
    }
    [[nodiscard]] const ExprPtr &GetCondition() const { return condition_; }
    [[nodiscard]] bool InElseBranch() const { return inElseBranch_; }
    [[nodiscard]] const std::vector<StmtPtr> &GetElseStmts() const { return elseStmts_; }
    [[nodiscard]] const std::vector<VarPtr> &GetReturnVars() const { return returnVars_; }

private:
    ExprPtr condition_;
    bool inElseBranch_ = false;
    std::vector<StmtPtr> elseStmts_;
    std::vector<VarPtr> returnVars_;
};

/**
 * \brief Context for building a program
 */
class ProgramContext : public BuildContext {
public:
    ProgramContext(std::string name, Span span)
        : BuildContext(Type::PROGRAM, std::move(span)), name_(std::move(name)) {}

    /**
     * \brief Declare a function and get its GlobalVar
     *
     * \param funcName Function name to declare
     * \return GlobalVar for the function
     */
    GlobalVarPtr DeclareFunction(const std::string &funcName);

    /**
     * \brief Get a GlobalVar for a declared function
     *
     * \param funcName Function name
     * \return GlobalVar for the function, or nullptr if not found
     */
    [[nodiscard]] GlobalVarPtr GetGlobalVar(const std::string &funcName) const;

    /**
     * \brief Add a function to the program
     *
     * \param func Function to add
     */
    void AddFunction(const FunctionPtr &func);

    /**
     * \brief Get the program name
     *
     * \return Program name
     */
    [[nodiscard]] const std::string &GetName() const { return name_; }

    /**
     * \brief Get all functions in the program
     *
     * \return Vector of functions
     */
    [[nodiscard]] const std::vector<FunctionPtr> &GetFunctions() const { return functions_; }

    /**
     * \brief Get all GlobalVars in the program
     *
     * \return Map of function names to GlobalVars
     */
    [[nodiscard]] const std::map<std::string, GlobalVarPtr> &GetGlobalVars() const { return globalVars_; }

    /**
     * \brief Get return types for a function by its GlobalVar
     *
     * \param gvar GlobalVar for the function
     * \return Vector of return types, or empty vector if function not yet added
     */
    [[nodiscard]] std::vector<TypePtr> GetReturnTypes(const GlobalVarPtr &gvar) const;

    // ProgramContext doesn't accumulate statements
    void AddStmt(const StmtPtr & /*stmt*/) override {
        throw InternalError("Cannot add statements directly to program context");
    }

private:
    std::string name_;
    std::vector<FunctionPtr> functions_;
    std::map<std::string, GlobalVarPtr> globalVars_;          // Track GlobalVars for cross-function calls
    std::map<std::string, std::vector<TypePtr>> returnTypes_; // Track return types for each function
};

} // namespace ir
} // namespace pypto
