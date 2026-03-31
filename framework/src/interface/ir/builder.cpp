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

#include "ir/builder.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/logging.h"

namespace pypto {
namespace ir {

// ========== IRBuilder Implementation ==========

IRBuilder::IRBuilder() = default;

IRBuilder::~IRBuilder() = default;

// ========== Function Building ==========

void IRBuilder::BeginFunction(const std::string &name, const Span &span, FunctionType type) {
    if (InFunction()) {
        throw RuntimeError("Cannot begin function '" + name + "': already inside function '" +
                           static_cast<FunctionContext *>(CurrentContext())->GetName() + "' at " +
                           CurrentContext()->GetBeginSpan().ToString());
    }

    contextStack_.push_back(std::make_unique<FunctionContext>(name, span, type));
}

VarPtr IRBuilder::FuncArg(const std::string &name, const TypePtr &type, const Span &span) {
    ValidateInFunction("FuncArg");

    auto var = std::make_shared<ir::Var>(name, type, span);
    static_cast<FunctionContext *>(CurrentContext())->AddParam(var);
    return var;
}

void IRBuilder::ReturnType(const TypePtr &type) {
    ValidateInFunction("ReturnType");
    static_cast<FunctionContext *>(CurrentContext())->AddReturnType(type);
}

FunctionPtr IRBuilder::EndFunction(const Span &endSpan) {
    ValidateInFunction("EndFunction");

    auto *funcCtx = static_cast<FunctionContext *>(CurrentContext());

    // Build body from accumulated statements
    StmtPtr body;
    const auto &stmts = funcCtx->GetStmts();
    if (stmts.empty()) {
        // Empty body - create empty SeqStmts
        body = std::make_shared<SeqStmts>(std::vector<StmtPtr>(), endSpan);
    } else if (stmts.size() == 1) {
        body = stmts[0];
    } else {
        body = std::make_shared<SeqStmts>(stmts, endSpan);
    }

    // Combine begin and end spans
    const Span &beginSpan = funcCtx->GetBeginSpan();
    Span combinedSpan(
        beginSpan.filename_, beginSpan.beginLine_, beginSpan.beginColumn_, endSpan.beginLine_, endSpan.beginColumn_);

    // Create function
    auto func = std::make_shared<Function>(funcCtx->GetName(), funcCtx->GetParams(), funcCtx->GetReturnTypes(), body,
        combinedSpan, funcCtx->GetFuncType());

    // Pop context
    contextStack_.pop_back();

    return func;
}

// ========== For Loop Building ==========

ForLoopBuilder IRBuilder::CreateForLoop(
    const VarPtr &loopVar, const ExprPtr &start, const ExprPtr &stop, const ExprPtr &step, const Span &span) {
    return ForLoopBuilder(loopVar, start, stop, step, span);
}

// ========== If Statement Building ==========

void IRBuilder::BeginIf(const ExprPtr &condition, const Span &span) {
    INTERNAL_CHECK(!contextStack_.empty())
        << "Cannot begin if statement: not inside a function or another valid context at " << span.ToString();
    contextStack_.push_back(std::make_unique<IfStmtContext>(condition, span));
}

void IRBuilder::BeginElse(const Span &span) {
    ValidateInIf("BeginElse");

    auto *ifCtx = static_cast<IfStmtContext *>(CurrentContext());
    INTERNAL_CHECK(!ifCtx->InElseBranch()) << "Cannot begin else branch: already in else branch at " << span.ToString();

    ifCtx->BeginElseBranch();
}

void IRBuilder::AddIfReturnVar(const VarPtr &var) {
    ValidateInIf("AddIfReturnVar");
    static_cast<IfStmtContext *>(CurrentContext())->AddReturnVar(var);
}

StmtPtr IRBuilder::EndIf(const Span &endSpan) {
    ValidateInIf("EndIf");

    auto *ifCtx = static_cast<IfStmtContext *>(CurrentContext());

    // Build then body
    StmtPtr thenBody;
    const auto &thenStmts = ifCtx->GetStmts();
    if (thenStmts.empty()) {
        thenBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>(), endSpan);
    } else if (thenStmts.size() == 1) {
        thenBody = thenStmts[0];
    } else {
        thenBody = std::make_shared<SeqStmts>(thenStmts, endSpan);
    }

    // Build else body (optional)
    std::optional<StmtPtr> elseBody;
    if (ifCtx->InElseBranch()) {
        const auto &elseStmts = ifCtx->GetElseStmts();
        if (!elseStmts.empty()) {
            if (elseStmts.size() == 1) {
                elseBody = elseStmts[0];
            } else {
                elseBody = std::make_shared<SeqStmts>(elseStmts, endSpan);
            }
        }
    }

    // Combine begin and end spans
    const Span &beginSpan = ifCtx->GetBeginSpan();
    Span combinedSpan(
        beginSpan.filename_, beginSpan.beginLine_, beginSpan.beginColumn_, endSpan.beginLine_, endSpan.beginColumn_);

    // Create if statement
    auto ifStmt =
        std::make_shared<IfStmt>(ifCtx->GetCondition(), thenBody, elseBody, ifCtx->GetReturnVars(), combinedSpan);

    // Pop context
    contextStack_.pop_back();

    // Emit to parent context if it exists
    if (!contextStack_.empty()) {
        CurrentContext()->AddStmt(ifStmt);
    }

    return ifStmt;
}

// ========== Program Building ==========

void IRBuilder::BeginProgram(const std::string &name, const Span &span) {
    if (InProgram()) {
        throw RuntimeError("Cannot begin program '" + name + "': already inside program '" +
                           static_cast<ProgramContext *>(CurrentContext())->GetName() + "' at " +
                           CurrentContext()->GetBeginSpan().ToString());
    }

    contextStack_.push_back(std::make_unique<ProgramContext>(name, span));
}

GlobalVarPtr IRBuilder::DeclareFunction(const std::string &funcName) {
    ValidateInProgram("DeclareFunction");
    return static_cast<ProgramContext *>(CurrentContext())->DeclareFunction(funcName);
}

GlobalVarPtr IRBuilder::GetGlobalVar(const std::string &funcName) {
    ValidateInProgram("GetGlobalVar");
    auto gvar = static_cast<ProgramContext *>(CurrentContext())->GetGlobalVar(funcName);
    if (!gvar) {
        throw RuntimeError("Function '" + funcName + "' not declared in current program");
    }
    return gvar;
}

void IRBuilder::AddFunction(const FunctionPtr &func) {
    ValidateInProgram("AddFunction");
    static_cast<ProgramContext *>(CurrentContext())->AddFunction(func);
}

ProgramPtr IRBuilder::EndProgram(const Span &endSpan) {
    ValidateInProgram("EndProgram");

    auto *progCtx = static_cast<ProgramContext *>(CurrentContext());

    // Combine begin and end spans
    const Span &beginSpan = progCtx->GetBeginSpan();
    Span combinedSpan(
        beginSpan.filename_, beginSpan.beginLine_, beginSpan.beginColumn_, endSpan.beginLine_, endSpan.beginColumn_);

    // Create program from functions vector
    auto program = std::make_shared<Program>(progCtx->GetFunctions(), progCtx->GetName(), combinedSpan);

    // Pop context
    contextStack_.pop_back();

    return program;
}

bool IRBuilder::InProgram() const {
    for (const auto &ctx : contextStack_) {
        if (ctx->GetType() == BuildContext::Type::PROGRAM) {
            return true;
        }
    }
    return false;
}

std::vector<TypePtr> IRBuilder::GetFunctionReturnTypes(const GlobalVarPtr &gvar) const {
    // Find the program context in the stack
    for (const auto &ctx : contextStack_) {
        if (ctx->GetType() == BuildContext::Type::PROGRAM) {
            auto *progCtx = static_cast<const ProgramContext *>(ctx.get());
            return progCtx->GetReturnTypes(gvar);
        }
    }
    return {};
}

// ========== Statement Recording ==========

void IRBuilder::Emit(const StmtPtr &stmt) {
    if (contextStack_.empty()) {
        throw RuntimeError("Cannot emit statement: not inside any context");
    }

    auto *ctx = CurrentContext();
    ctx->AddStmt(stmt);
}

AssignStmtPtr IRBuilder::Assign(const VarPtr &var, const ExprPtr &value, const Span &span) {
    auto assign = std::make_shared<AssignStmt>(var, value, span);
    Emit(assign);
    return assign;
}

VarPtr IRBuilder::Var(const std::string &name, const TypePtr &type, const Span &span) {
    return std::make_shared<ir::Var>(name, type, span);
}

ReturnStmtPtr IRBuilder::Return(const std::vector<ExprPtr> &values, const Span &span) {
    auto returnStmt = std::make_shared<ReturnStmt>(values, span);
    Emit(returnStmt);
    return returnStmt;
}

ReturnStmtPtr IRBuilder::Return(const Span &span) {
    auto returnStmt = std::make_shared<ReturnStmt>(span);
    Emit(returnStmt);
    return returnStmt;
}

// ========== Context State Queries ==========

BuildContext *IRBuilder::CurrentContext() {
    if (contextStack_.empty()) {
        return nullptr;
    }
    return contextStack_.back().get();
}

bool IRBuilder::InFunction() const {
    for (const auto &ctx : contextStack_) {
        if (ctx->GetType() == BuildContext::Type::FUNCTION) {
            return true;
        }
    }
    return false;
}

bool IRBuilder::InIf() const {
    if (contextStack_.empty()) {
        return false;
    }
    return contextStack_.back()->GetType() == BuildContext::Type::IF_STMT;
}

// ========== Private Helpers ==========

template <typename T>
T *IRBuilder::GetCurrentContextAs() {
    auto *ctx = CurrentContext();
    if (!ctx) {
        return nullptr;
    }
    return dynamic_cast<T *>(ctx);
}

void IRBuilder::ValidateInFunction(const std::string &operation) {
    INTERNAL_CHECK(InFunction()) << operation << " can only be called inside a function context";
    INTERNAL_CHECK(CurrentContext()->GetType() == BuildContext::Type::FUNCTION)
        << operation << " must be called directly in function context, not nested";
}

void IRBuilder::ValidateInIf(const std::string &operation) {
    INTERNAL_CHECK(InIf()) << operation << " can only be called inside an if statement context";
}

void IRBuilder::ValidateInProgram(const std::string &operation) {
    INTERNAL_CHECK(InProgram()) << operation << " can only be called inside a program context";
}

// ========== ForLoopBuilder Implementation ==========

StmtPtr ForLoopBuilder::Build(const Span &endSpan) {
    if (iterArgs_.size() != returnVars_.size()) {
        std::ostringstream oss;
        oss << "For loop has " << iterArgs_.size() << " iteration arguments but " << returnVars_.size()
            << " return variables. They must match.";
        throw RuntimeError(oss.str());
    }

    // Build body from accumulated statements
    StmtPtr body;
    if (stmts_.empty()) {
        body = std::make_shared<SeqStmts>(std::vector<StmtPtr>(), endSpan);
    } else if (stmts_.size() == 1) {
        body = stmts_[0];
    } else {
        body = std::make_shared<SeqStmts>(stmts_, endSpan);
    }

    // Combine begin and end spans
    Span combinedSpan(
        beginSpan_.filename_, beginSpan_.beginLine_, beginSpan_.beginColumn_, endSpan.beginLine_, endSpan.beginColumn_);

    return std::make_shared<ForStmt>(loopVar_, start_, stop_, step_, iterArgs_, body, returnVars_, combinedSpan);
}

// ========== ProgramContext Implementation ==========

GlobalVarPtr ProgramContext::DeclareFunction(const std::string &funcName) {
    // Check if already declared
    auto it = globalVars_.find(funcName);
    if (it != globalVars_.end()) {
        return it->second;
    }

    // Create new GlobalVar
    auto gvar = std::make_shared<GlobalVar>(funcName);
    globalVars_[funcName] = gvar;
    return gvar;
}

GlobalVarPtr ProgramContext::GetGlobalVar(const std::string &funcName) const {
    auto it = globalVars_.find(funcName);
    if (it != globalVars_.end()) {
        return it->second;
    }
    return nullptr;
}

void ProgramContext::AddFunction(const FunctionPtr &func) {
    INTERNAL_CHECK(func) << "Cannot add null function to program";

    // Verify function was declared (if not, declare it automatically)
    auto it = globalVars_.find(func->name_);
    if (it == globalVars_.end()) {
        // Function wasn't declared, declare it now for convenience
        DeclareFunction(func->name_);
    }

    // Store return types for this function
    returnTypes_[func->name_] = func->returnTypes_;

    functions_.push_back(func);
}

std::vector<TypePtr> ProgramContext::GetReturnTypes(const GlobalVarPtr &gvar) const {
    auto it = returnTypes_.find(gvar->name_);
    if (it != returnTypes_.end()) {
        return it->second;
    }
    return {};
}

} // namespace ir
} // namespace pypto
