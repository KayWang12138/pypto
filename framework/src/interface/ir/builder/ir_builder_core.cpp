#include "ir/builder/ir_builder.h"

#include <stdexcept>
#include <utility>

namespace pto {

IRBuilder::IRBuilder(std::shared_ptr<ProgramModule> module) : module_(module) {}

void IRBuilder::SetModule(std::shared_ptr<ProgramModule> m) {
    module_ = m;
}
    
std::shared_ptr<Function> IRBuilder::CreateFunction(
    std::string name,
    FunctionKind kind,
    FunctionSignature sig,
    bool setAsEntry) {

    if (!module_) {
        throw std::runtime_error("IRBuilder::CreateFunction: module is null");
    }

    auto fn = std::make_shared<Function>(std::move(name), kind, std::move(sig));
    module_->AddFunction(fn);

    if (setAsEntry) {
        module_->SetProgramEntry(fn);
    }

    return fn;
}

OpStatementPtr IRBuilder::GetOrCreateActiveOpStmt() {
    if (!compound_) throw std::runtime_error("IRBuilder::GetOrCreateActiveOpStmt: compound is null");
    if (opStmt_) return opStmt_;

    // Create a new op statement at current scope tail
    auto opStmt = std::make_shared<OpStatement>();
    compound_->AddStatement(opStmt);
    opStmt_ = opStmt;
    return opStmt;
}
} // namespace pto
