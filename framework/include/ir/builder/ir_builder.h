// ir/builder/ir_builder.h
// PTO-IR prototype: minimal IRBuilder (insertion + scope + emit).
// All comments must remain in English for consistency.

#pragma once

#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/op/op_opcode.h"
#include "ir/op/op_payload.h"
#include "ir/value.h"
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
    explicit IRBuilder(std::shared_ptr<ProgramModule> module);

    // ===== Module / Function =====
    void SetModule(std::shared_ptr<ProgramModule> m);

    // ===== Function =====

    std::shared_ptr<Function> CreateFunction(
        std::string name,
        FunctionKind kind,
        FunctionSignature sig,
        bool setAsEntry = false);

    std::shared_ptr<Function> GetCurrentFunction() const { return func_; }
    std::shared_ptr<CompoundStatement> GetCurrentCompound() const { return compound_; }
    std::shared_ptr<OpStatement> GetCurrentOpStmt() const { return opStmt_; }

    // ===== Insertion point =====
    OpStatementPtr GetOrCreateActiveOpStmt();
    friend class ScopeGuard;

    // ===== Scope registration =====
    ValuePtr AddToCompound(ValuePtr v);

    // Optional convenience: create values (not "like", just explicit)
    std::shared_ptr<Tensor> CreateTensor(const std::vector<Scalar>& shape, DataType dt, std::string name = "");
    std::shared_ptr<Tile>   CreateTile(const std::vector<size_t>& shape, DataType dt, std::string name = "");
    std::shared_ptr<Scalar> CreateScalar(DataType dt, std::string name = "");
    std::shared_ptr<Scalar> CreateConst(int64_t v, std::string name = "");
    std::shared_ptr<Scalar> CreateConst(double v, std::string name = "");

    // ===== Emit op (used by schema build) =====
    OperationPtr Emit(OperationPtr op);

    // ===== The ONLY op-building entry =====
    // All semantics (results/payload rules) are in Schema/Trait (BuildBySchema).
    // Writeback-style: caller provides outputs (for a few ops like assemble)

    ValuePtrs CreateOp(Opcode opcode,
                    ValuePtrs inputs,
                    std::shared_ptr<OpPayload> payload = nullptr,
                    std::string name = "");

    // ===== Statement building (still belongs to IRBuilder) =====
    OpStatementPtr CreateOpStmt();

    ForStatementPtr CreateForStmt(std::shared_ptr<Scalar> iv,
                                std::shared_ptr<Scalar> start,
                                std::shared_ptr<Scalar> end,
                                std::shared_ptr<Scalar> step);

    IfStatementPtr CreateIfStmt(std::string cond);

    YieldStatementPtr CreateYield(ValuePtrs values);

    ReturnStatementPtr CreateReturn(ValuePtrs values);

    // Enter nested scopes
    std::shared_ptr<ScopeGuard> EnterFunctionBody(std::shared_ptr<Function> func);
    std::shared_ptr<ScopeGuard> EnterForBody(ForStatementPtr st);
    std::shared_ptr<ScopeGuard> EnterIfThen(IfStatementPtr st);
    std::shared_ptr<ScopeGuard> EnterIfElse(IfStatementPtr st);
    void ExitIfStatement(IfStatementPtr st);
    void ExitForStatement(ForStatementPtr st);

private:
    std::shared_ptr<ProgramModule> module_{nullptr};
    std::shared_ptr<Function> func_{nullptr};
    std::shared_ptr<CompoundStatement> compound_{nullptr};
    std::shared_ptr<OpStatement> opStmt_{nullptr};
};

} // namespace pto