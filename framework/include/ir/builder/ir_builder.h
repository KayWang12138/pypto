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

    void SetCurrentFunction(Function& f);

    Function* GetCurrentFunction() const { return func_; }
    Scope* GetCurrentScope() const { return scope_; }
    BlockStatement* GetCurrentBlock() const { return block_; }

    // ===== Insertion point =====
    BlockStatement& GetOrCreateActiveBlock();
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
    BlockStatement& CreateBlockStmt();

    ReturnStatement& CreateReturn(ValuePtrs values);

    // Enter nested scopes (optional sugar)
    std::shared_ptr<ScopeGuard> EnterFunctionBody(Function& func);

private:
    ProgramModule* module_{nullptr};
    Function* func_{nullptr};
    Scope* scope_{nullptr};
    BlockStatement* block_{nullptr};
};

} // namespace pto