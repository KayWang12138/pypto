#include "pass/pass_manager.h"
#include "pass/dump_ir_pass.h"
#include "ir/program.h"
#include "ir/function.h"
#include <iostream>

namespace pto {

PassManager::PassManager() {
    // Add default passes
    // DumpIRPass with default output directory "./output"
    passes_.push_back(std::make_shared<DumpIRPass>("DumpIR", "./output"));
}

Result PassManager::Run(ProgramModule& module, bool print) {
    for (const auto& pass : passes_) {
        auto result = pass->Run(module);
        if (print && pass != passes_[0]) {
            std::static_pointer_cast<DumpIRPass>(passes_[0])->DumpIRAfterPass(pass->GetName(), module);
        }
        if (result == Result::Failure) {
            return Result::Failure;
        }
    }
    return Result::Success;
}

Result PassManager::Run(std::shared_ptr<Function> func, bool print) {
    for (const auto& pass : passes_) {
        auto result = pass->RunOnFunction(func);
        if (print) {
            std::cout << "============= after pass: " << pass->GetName() 
                << " ===============" << std::endl;
            std::cout << *func << std::endl << std::endl;
        }
        if (result == Result::Failure) {
            std::cout << "!!! pass: " << pass->GetName() 
                << " RunOnFunction fail, func name: " << func->GetName() 
                << " !!!" << std::endl;
            return Result::Failure;
        }
    }
    return Result::Success;
}

}  // namespace pto

