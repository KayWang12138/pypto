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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ir/core.h"
#include "ir/expr.h"
#include "ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declarations for friend classes
class IRVisitor;
class IRMutator;

enum class ForKind : uint8_t {
    Sequential = 0,
    Parallel = 1,
    Unroll = 2
};

enum class ChunkPolicy : uint8_t {
    LeadingFull = 0
};

enum class LoopOrigin : uint8_t {
    Original = 0,
    ChunkOuter = 1,
    ChunkInner = 2,
    ChunkRemainder = 3
};

enum class ScopeKind : uint8_t {
    InCore = 0
};

enum class SectionKind : uint8_t {
    Vector = 0,
    Cube = 1
};

inline std::string ForKindToString(ForKind kind)
{
    switch (kind) {
        case ForKind::Sequential: return "Sequential";
        case ForKind::Parallel: return "Parallel";
        case ForKind::Unroll: return "Unroll";
        default: break;
    }
    throw TypeError("Unknown ForKind");
}

inline ForKind StringToForKind(const std::string& str)
{
    if (str == "Sequential") return ForKind::Sequential;
    if (str == "Parallel") return ForKind::Parallel;
    if (str == "Unroll") return ForKind::Unroll;
    throw TypeError("Unknown ForKind: " + str);
}

inline std::string ChunkPolicyToString(ChunkPolicy policy)
{
    switch (policy) {
        case ChunkPolicy::LeadingFull: return "LeadingFull";
        default: break;
    }
    throw TypeError("Unknown ChunkPolicy");
}

inline ChunkPolicy StringToChunkPolicy(const std::string& str)
{
    if (str == "LeadingFull" || str == "leading_full") return ChunkPolicy::LeadingFull;
    throw TypeError("Unknown ChunkPolicy: " + str);
}

inline std::string LoopOriginToString(LoopOrigin origin)
{
    switch (origin) {
        case LoopOrigin::Original: return "Original";
        case LoopOrigin::ChunkOuter: return "ChunkOuter";
        case LoopOrigin::ChunkInner: return "ChunkInner";
        case LoopOrigin::ChunkRemainder: return "ChunkRemainder";
        default: break;
    }
    throw TypeError("Unknown LoopOrigin");
}

inline LoopOrigin StringToLoopOrigin(const std::string& str)
{
    if (str == "Original") return LoopOrigin::Original;
    if (str == "ChunkOuter") return LoopOrigin::ChunkOuter;
    if (str == "ChunkInner") return LoopOrigin::ChunkInner;
    if (str == "ChunkRemainder") return LoopOrigin::ChunkRemainder;
    throw TypeError("Unknown LoopOrigin: " + str);
}

inline std::string ScopeKindToString(ScopeKind kind)
{
    switch (kind) {
        case ScopeKind::InCore: return "InCore";
        default: break;
    }
    throw TypeError("Unknown ScopeKind");
}

inline ScopeKind StringToScopeKind(const std::string& str)
{
    if (str == "InCore") return ScopeKind::InCore;
    throw TypeError("Unknown ScopeKind: " + str);
}

inline std::string SectionKindToString(SectionKind kind)
{
    switch (kind) {
        case SectionKind::Vector: return "Vector";
        case SectionKind::Cube: return "Cube";
        default: break;
    }
    throw TypeError("Unknown SectionKind");
}

inline SectionKind StringToSectionKind(const std::string& str)
{
    if (str == "Vector") return SectionKind::Vector;
    if (str == "Cube") return SectionKind::Cube;
    throw TypeError("Unknown SectionKind: " + str);
}

/**
 * \brief Base class for all statements in the IR
 *
 * Statements represent operations that perform side effects or control flow.
 * All statements are immutable.
 */
class Stmt : public IRNode {
public:
    /**
     * \brief Create a statement
     *
     * \param span Source location
     */
    explicit Stmt(Span s) : IRNode(std::move(s)) {}
    ~Stmt() override = default;

    /**
     * \brief Get the type name of this statement
     *
     * \return Human-readable type name (e.g., "Stmt", "Assign", "Return")
     */
    [[nodiscard]] std::string TypeName() const override { return "Stmt"; }

    static constexpr auto GetFieldDescriptors() { return IRNode::GetFieldDescriptors(); }
};

using StmtPtr = std::shared_ptr<const Stmt>;

/**
 * \brief Assignment statement
 *
 * Represents an assignment operation: var = value
 * where var is a variable and value is an expression.
 */
class AssignStmt : public Stmt {
public:
    VarPtr var_;    // Variable
    ExprPtr value_; // Expression

    /**
     * \brief Create an assignment statement
     *
     * \param var Variable
     * \param value Expression
     * \param span Source location
     */
    AssignStmt(VarPtr var, ExprPtr value, Span span)
        : Stmt(std::move(span)), var_(std::move(var)), value_(std::move(value))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::AssignStmt; }
    [[nodiscard]] std::string TypeName() const override { return "AssignStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (var and value as DEF and USUAL fields)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(),
            std::make_tuple(
                reflection::DefField(&AssignStmt::var_, "var"), reflection::UsualField(&AssignStmt::value_, "value")));
    }
};

using AssignStmtPtr = std::shared_ptr<const AssignStmt>;

/**
 * \brief Conditional statement
 *
 * Represents an if-else statement: if condition then then_body else else_body
 * where condition is an expression and then_body/else_body is statement.
 */
class IfStmt : public Stmt {
public:
    /**
     * \brief Create a conditional statement with then and else branches
     *
     * \param condition Condition expression
     * \param thenBody Then branch statement
     * \param elseBody Else branch statement (can be optional)
     * \param returnVars Return variables (can be empty)
     * \param span Source location
     */
    IfStmt(
        ExprPtr condition, StmtPtr thenBody, std::optional<StmtPtr> elseBody, std::vector<VarPtr> returnVars, Span span)
        : Stmt(std::move(span)),
          condition_(std::move(condition)),
          thenBody_(std::move(thenBody)),
          elseBody_(std::move(elseBody)),
          returnVars_(std::move(returnVars))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::IfStmt; }
    [[nodiscard]] std::string TypeName() const override { return "IfStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (condition, then_body, else_body as USUAL fields)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(), std::make_tuple(
                                             reflection::UsualField(&IfStmt::condition_, "condition"),
                                             reflection::UsualField(&IfStmt::thenBody_, "then_body"),
                                             reflection::UsualField(&IfStmt::elseBody_, "else_body"),
                                             reflection::DefField(&IfStmt::returnVars_, "return_vars")));
    }

public:
    ExprPtr condition_;               // Condition expression
    StmtPtr thenBody_;                // Then branch statement
    std::optional<StmtPtr> elseBody_; // Else branch statement (optional)
    std::vector<VarPtr> returnVars_;  // Return variables (can be empty)
};

using IfStmtPtr = std::shared_ptr<const IfStmt>;

/**
 * \brief Yield statement
 *
 * Represents a yield operation: yield value
 * where value is a list of variables to yield.
 */
class YieldStmt : public Stmt {
public:
    /**
     * \brief Create a yield statement
     *
     * \param value List of variables to yield (can be empty)
     * \param span Source location
     */
    YieldStmt(std::vector<ExprPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}

    /**
     * \brief Create a yield statement without values
     *
     * \param span Source location
     */
    explicit YieldStmt(Span span) : Stmt(std::move(span)), value_() {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::YieldStmt; }
    [[nodiscard]] std::string TypeName() const override { return "YieldStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (value as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(), std::make_tuple(reflection::UsualField(&YieldStmt::value_, "value")));
    }

public:
    std::vector<ExprPtr> value_; // List of expressions to yield
};

using YieldStmtPtr = std::shared_ptr<const YieldStmt>;

/**
 * \brief Return statement
 *
 * Represents a return operation: return value
 * where value is a list of expressions to return.
 */
class ReturnStmt : public Stmt {
public:
    /**
     * \brief Create a return statement
     *
     * \param value List of expressions to return (can be empty)
     * \param span Source location
     */
    ReturnStmt(std::vector<ExprPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}

    /**
     * \brief Create a return statement without values
     *
     * \param span Source location
     */
    explicit ReturnStmt(Span span) : Stmt(std::move(span)), value_() {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ReturnStmt; }
    [[nodiscard]] std::string TypeName() const override { return "ReturnStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (value as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(), std::make_tuple(reflection::UsualField(&ReturnStmt::value_, "value")));
    }

public:
    std::vector<ExprPtr> value_; // List of expressions to return
};

using ReturnStmtPtr = std::shared_ptr<const ReturnStmt>;

/**
 * \brief For loop statement
 *
 * Represents a for loop with optional loop-carried values (SSA-style iteration).
 *
 * **Basic loop:** for loop_var in range(start, stop, step): body
 *
 * **Loop with iteration arguments:**
 * for loop_var, (iter_arg1, iter_arg2) in pl.range(start, stop, step, init_values=[...]):
 *     iter_arg1, iter_arg2 = pl.yield_(new_val1, new_val2)
 * return_var1 = iter_arg1
 * return_var2 = iter_arg2
 *
 * **Key Relationships:**
 * - iter_args: IterArg variables scoped to loop body, carry values between iterations
 * - return_vars: Var variables that capture final iteration values, accessible after loop
 * - Number of iter_args must equal number of return_vars
 * - Number of yielded values must equal number of iter_args
 * - IterArgs cannot be directly accessed outside the loop; use return_vars instead
 */
class ForStmt : public Stmt {
public:
    ForStmt(
        VarPtr loopVar, ExprPtr start, ExprPtr stop, ExprPtr step, std::vector<IterArgPtr> iterArgs, StmtPtr body,
        std::vector<VarPtr> returnVars, Span span, ForKind kind = ForKind::Sequential,
        std::optional<ExprPtr> chunkSize = std::nullopt, ChunkPolicy chunkPolicy = ChunkPolicy::LeadingFull,
        LoopOrigin loopOrigin = LoopOrigin::Original)
        : Stmt(std::move(span)),
          loopVar_(std::move(loopVar)),
          start_(std::move(start)),
          stop_(std::move(stop)),
          step_(std::move(step)),
          iterArgs_(std::move(iterArgs)),
          body_(std::move(body)),
          returnVars_(std::move(returnVars)),
          kind_(kind),
          chunkSize_(std::move(chunkSize)),
          chunkPolicy_(chunkPolicy),
          loopOrigin_(loopOrigin)
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ForStmt; }
    [[nodiscard]] std::string TypeName() const override { return "ForStmt"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(),
            std::make_tuple(
                reflection::DefField(&ForStmt::loopVar_, "loop_var"), reflection::UsualField(&ForStmt::start_, "start"),
                reflection::UsualField(&ForStmt::stop_, "stop"), reflection::UsualField(&ForStmt::step_, "step"),
                reflection::DefField(&ForStmt::iterArgs_, "iter_args"), reflection::UsualField(&ForStmt::body_, "body"),
                reflection::DefField(&ForStmt::returnVars_, "return_vars"),
                reflection::UsualField(&ForStmt::kind_, "kind"),
                reflection::UsualField(&ForStmt::chunkSize_, "chunk_size"),
                reflection::UsualField(&ForStmt::chunkPolicy_, "chunk_policy"),
                reflection::IgnoreField(&ForStmt::loopOrigin_, "loop_origin")));
    }

public:
    VarPtr loopVar_;
    ExprPtr start_;
    ExprPtr stop_;
    ExprPtr step_;
    std::vector<IterArgPtr> iterArgs_;
    StmtPtr body_;
    std::vector<VarPtr> returnVars_;
    ForKind kind_ = ForKind::Sequential;
    std::optional<ExprPtr> chunkSize_;
    ChunkPolicy chunkPolicy_ = ChunkPolicy::LeadingFull;
    LoopOrigin loopOrigin_ = LoopOrigin::Original;
};

using ForStmtPtr = std::shared_ptr<const ForStmt>;

/**
 * \brief While loop statement
 *
 * Represents a while loop with optional loop-carried values (SSA-style iteration).
 *
 * **Basic loop:** while condition: body
 *
 * **Loop with iteration arguments:**
 * while condition, (iter_arg1, iter_arg2) with init_values=[...]:
 *     iter_arg1, iter_arg2 = pl.yield_(new_val1, new_val2)
 * return_var1 = iter_arg1
 * return_var2 = iter_arg2
 *
 * **Key Relationships:**
 * - condition: Boolean expression evaluated each iteration using current iter_args
 * - iter_args: IterArg variables scoped to loop body, carry values between iterations
 * - return_vars: Var variables that capture final iteration values, accessible after loop
 * - Number of iter_args must equal number of return_vars
 * - Number of yielded values must equal number of iter_args
 */
class WhileStmt : public Stmt {
public:
    /**
     * \brief Create a while loop statement
     *
     * \param condition Boolean condition expression
     * \param iterArgs Iteration arguments (loop-carried values, scoped to loop body)
     * \param body Loop body statement (must yield values matching iterArgs if non-empty)
     * \param returnVars Return variables (capture final values, accessible after loop)
     * \param span Source location
     */
    WhileStmt(
        ExprPtr condition, std::vector<IterArgPtr> iterArgs, StmtPtr body, std::vector<VarPtr> returnVars, Span span)
        : Stmt(std::move(span)),
          condition_(std::move(condition)),
          iterArgs_(std::move(iterArgs)),
          body_(std::move(body)),
          returnVars_(std::move(returnVars))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::WhileStmt; }
    [[nodiscard]] std::string TypeName() const override { return "WhileStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (iter_args as DEF, condition as USUAL, body as USUAL, return_vars as
     * DEF). Iter args must be visited before condition/body so structural comparison can bind loop-carried vars first.
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(), std::make_tuple(
                                             reflection::DefField(&WhileStmt::iterArgs_, "iter_args"),
                                             reflection::UsualField(&WhileStmt::condition_, "condition"),
                                             reflection::UsualField(&WhileStmt::body_, "body"),
                                             reflection::DefField(&WhileStmt::returnVars_, "return_vars")));
    }

public:
    ExprPtr condition_;                // Condition expression (evaluated each iteration)
    std::vector<IterArgPtr> iterArgs_; // Loop-carried values (scoped to loop body)
    StmtPtr body_;                     // Loop body statement (must yield if iter_args non-empty)
    std::vector<VarPtr> returnVars_;   // Variables capturing final iteration values (accessible after loop)
};

using WhileStmtPtr = std::shared_ptr<const WhileStmt>;

/**
 * \brief Sequence of statements
 *
 * Represents a sequence of statements: stmt1; stmt2; ... stmtN
 * where stmts is a list of statements.
 */
class SeqStmts : public Stmt {
public:
    /**
     * \brief Create a sequence of statements
     *
     * \param stmts List of statements
     * \param span Source location
     */
    SeqStmts(std::vector<StmtPtr> stmts, Span span) : Stmt(std::move(span)), stmts_(std::move(stmts)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::SeqStmts; }
    [[nodiscard]] std::string TypeName() const override { return "SeqStmts"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (stmts as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(), std::make_tuple(reflection::UsualField(&SeqStmts::stmts_, "stmts")));
    }

    /**
     * @brief Create a normalized statement from a list of statements
     *
     * Flattens nested SeqStmts and unwraps single-child sequences:
     * - Flatten({a, SeqStmts({b, c}), d}, span) → SeqStmts({a, b, c, d})
     * - Flatten({a}, span) → a
     * - Flatten({}, span) → SeqStmts({})
     */
    static StmtPtr Flatten(std::vector<StmtPtr> stmts, Span span)
    {
        std::vector<StmtPtr> flat;
        for (auto& s : stmts) {
            if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(s)) {
                // Recursively flatten nested SeqStmts
                for (const auto& inner : seq->stmts_) {
                    if (auto inner_seq = std::dynamic_pointer_cast<const SeqStmts>(inner)) {
                        flat.insert(flat.end(), inner_seq->stmts_.begin(), inner_seq->stmts_.end());
                    } else {
                        flat.push_back(inner);
                    }
                }
            } else {
                flat.push_back(std::move(s));
            }
        }
        if (flat.size() == 1) {
            return flat[0];
        }
        return std::make_shared<SeqStmts>(std::move(flat), std::move(span));
    }

public:
    std::vector<StmtPtr> stmts_; // List of statements
};

using SeqStmtsPtr = std::shared_ptr<const SeqStmts>;

class ScopeStmt : public Stmt {
public:
    ScopeStmt(ScopeKind scopeKind, StmtPtr body, Span span)
        : Stmt(std::move(span)), scopeKind_(scopeKind), body_(std::move(body))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ScopeStmt; }
    [[nodiscard]] std::string TypeName() const override { return "ScopeStmt"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(),
            std::make_tuple(
                reflection::UsualField(&ScopeStmt::scopeKind_, "scope_kind"),
                reflection::UsualField(&ScopeStmt::body_, "body")));
    }

public:
    ScopeKind scopeKind_;
    StmtPtr body_;
};

using ScopeStmtPtr = std::shared_ptr<const ScopeStmt>;

class SectionStmt : public Stmt {
public:
    SectionStmt(SectionKind sectionKind, StmtPtr body, Span span)
        : Stmt(std::move(span)), sectionKind_(sectionKind), body_(std::move(body))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::SectionStmt; }
    [[nodiscard]] std::string TypeName() const override { return "SectionStmt"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(),
            std::make_tuple(
                reflection::UsualField(&SectionStmt::sectionKind_, "section_kind"),
                reflection::UsualField(&SectionStmt::body_, "body")));
    }

public:
    SectionKind sectionKind_;
    StmtPtr body_;
};

using SectionStmtPtr = std::shared_ptr<const SectionStmt>;

class OpStmts : public Stmt {
public:
    OpStmts(std::vector<StmtPtr> stmts, Span span);

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::OpStmts; }
    [[nodiscard]] std::string TypeName() const override { return "OpStmts"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(), std::make_tuple(reflection::UsualField(&OpStmts::stmts_, "stmts")));
    }

public:
    std::vector<StmtPtr> stmts_;
};

using OpStmtsPtr = std::shared_ptr<const OpStmts>;

/**
 * \brief Evaluation statement
 *
 * Represents an expression executed as a statement: expr
 * where expr is an expression (typically a Call).
 * This is used for expressions that have side effects but no return value
 * (or return value is ignored).
 */
class EvalStmt : public Stmt {
public:
    /**
     * \brief Create an evaluation statement
     *
     * \param expr Expression to execute
     * \param span Source location
     */
    EvalStmt(ExprPtr expr, Span span) : Stmt(std::move(span)), expr_(std::move(expr)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::EvalStmt; }
    [[nodiscard]] std::string TypeName() const override { return "EvalStmt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (expr as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(), std::make_tuple(reflection::UsualField(&EvalStmt::expr_, "expr")));
    }

public:
    ExprPtr expr_; // Expression
};

using EvalStmtPtr = std::shared_ptr<const EvalStmt>;

/**
 * \brief Break statement
 *
 * Represents a break statement used to exit a loop.
 */
class BreakStmt : public Stmt {
public:
    explicit BreakStmt(Span span) : Stmt(std::move(span)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::BreakStmt; }
    [[nodiscard]] std::string TypeName() const override { return "BreakStmt"; }

    static constexpr auto GetFieldDescriptors() { return Stmt::GetFieldDescriptors(); }
};

using BreakStmtPtr = std::shared_ptr<const BreakStmt>;

/**
 * \brief Continue statement
 *
 * Represents a continue statement used to skip to the next loop iteration.
 */
class ContinueStmt : public Stmt {
public:
    explicit ContinueStmt(Span span) : Stmt(std::move(span)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ContinueStmt; }
    [[nodiscard]] std::string TypeName() const override { return "ContinueStmt"; }

    static constexpr auto GetFieldDescriptors() { return Stmt::GetFieldDescriptors(); }
};

using ContinueStmtPtr = std::shared_ptr<const ContinueStmt>;

} // namespace ir
} // namespace pypto
