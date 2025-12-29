#include "pass/pass.h"
#include "ir/program.h"
#include <iostream>

namespace pto {

Result Pass::Run(ProgramModule &module) {
    std::cout << "run pass: " << name_ << std::endl;
    if (PreCheck(module) == Result::Failure) {
        std::cout << "!!! pass: " << GetName() << " PreCheck fail !!!" << std::endl; 
    }
    if (RunOnModule(module) == Result::Failure) {
        std::cout << "!!! pass: " << GetName() << " RunOnModule fail !!!" << std::endl; 
    }
    if (PostCheck(module) == Result::Failure) {
        std::cout << "!!! pass: " << GetName() << " PostCheck fail !!!" << std::endl; 
    }
    return Result::Success;
}

Result Pass::RunOnModule(ProgramModule &module) {
    for (auto funcPtr: module.GetFunctions()) {
        if (RunOnFunction(funcPtr) == Result::Failure) {
            std::cout << "!!! pass: " << GetName() << " RunOnFunction fail, func name: " << funcPtr->GetName() 
                << " !!!" << std::endl; 
            return Result::Failure;
        }
    }
    return Result::Success;
}

}  // namespace pto