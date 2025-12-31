// ir/builder/ir_builder.h
// PTO-IR prototype: minimal IRBuilder (insertion + scope + emit).
// All comments must remain in English for consistency.

#pragma once

#include "ir/program.h"
#include "ir/function.h"
#include "ir/scope.h"
#include "ir/statement.h"
#include "ir/op/op_opcode.h"
#include "ir/op/op_payload.h"
#include "ir/type.h"
#include "ir/builder/op_builder.h"
#include "ir/builder/guard.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace pto {

class IRBuilder {
public:
    IRBuilder() = default;
    explicit IRBuilder(ProgramModule* module);

    // ===== Module / Function =====
    void SetModule(ProgramModule& m);

    // ===== Function =====

    std::shared_ptr<Function> CreateFunction(
        std::string name,
        FunctionKind kind,
        FunctionSignature sig,
        bool setAsEntry = false);

    Function* GetCurrentFunction() const { return func_; }
    Scope* GetCurrentScope() const { return scope_; }
    OpStatement* GetCurrentOpStmt() const { return opStmt_; }

    // ===== Insertion point =====
    OpStatement& GetOrCreateActiveOpStmt();
    friend class ScopeGuard;

    // ===== Scope registration =====
    ValuePtr AddToScope(ValuePtr v);

    // Optional convenience: create values (not "like", just explicit)
    std::shared_ptr<Tensor> CreateTensor(const std::vector<Scalar>& shape, DataType dt, std::string name = "");
    std::shared_ptr<Tile>   CreateTile(const std::vector<size_t>& shape, DataType dt, std::string name = "");
    std::shared_ptr<Scalar> CreateScalar(DataType dt, std::string name = "");
    std::shared_ptr<Scalar> CreateConst(int64_t v, std::string name = "");
    std::shared_ptr<Scalar> CreateConst(double v, std::string name = "");

    // ===== Emit op (used by schema build) =====
    Operation& Emit(OperationPtr op);

    // ===== The ONLY op-building entry =====
    // All semantics (results/payload rules) are in Schema/Trait (BuildBySchema).
    // Writeback-style: caller provides outputs (for a few ops like assemble)

    ValuePtrs CreateOp(Opcode opcode,
                    ValuePtrs inputs,
                    std::shared_ptr<OpPayload> payload = nullptr,
                    std::string name = "");

    // ===== Statement building (still belongs to IRBuilder) =====
    OpStatement& CreateOpStmt();

    ForStatement& CreateForStmt(std::shared_ptr<Scalar> iv,
                                std::shared_ptr<Scalar> start,
                                std::shared_ptr<Scalar> end,
                                std::shared_ptr<Scalar> step);

    IfStatement& CreateIfStmt(std::string cond);

    YieldStatement& CreateYield(ValuePtrs values);

    ReturnStatement& CreateReturn(ValuePtrs values);

    // Enter nested scopes
    std::shared_ptr<ScopeGuard> EnterFunctionBody(Function& func);
    std::shared_ptr<ScopeGuard> EnterForBody(ForStatement& st);
    std::shared_ptr<ScopeGuard> EnterIfThen(IfStatement& st);
    std::shared_ptr<ScopeGuard> EnterIfElse(IfStatement& st);
    void ExitIfStatement(IfStatement& st);
    void ExitForStatement(ForStatement& st);

private:
    ProgramModule* module_{nullptr};
    Function* func_{nullptr};
    Scope* scope_{nullptr};
    OpStatement* opStmt_{nullptr};
};

} // namespace pto