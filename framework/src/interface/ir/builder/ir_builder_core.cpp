#include "ir/builder/ir_builder.h"

#include <stdexcept>
#include <utility>

namespace pto {

IRBuilder::IRBuilder(ProgramModule* module) : module_(module) {}

void IRBuilder::SetModule(ProgramModule& m) {
    module_ = &m;
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

BlockStatement& IRBuilder::GetOrCreateActiveBlock() {
    if (!scope_) throw std::runtime_error("IRBuilder::GetOrCreateActiveBlock: scope is null");
    if (block_) return *block_;

    // Create a new block statement at current scope tail
    auto blk = std::make_shared<BlockStatement>();
    auto& ref = *blk;
    scope_->AddStatement(std::move(blk));
    block_ = &ref;
    return ref;
}
} // namespace pto
