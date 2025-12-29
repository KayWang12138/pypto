#pragma once

#include "ir/function.h"
#include "ir/operation.h"
#include "ir/program.h"
#include "ir/statement.h"

#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace pto {

enum class Result {
    Success,
    Failure
};

class Pass {
public:
    Pass(const std::string& name) : name_(name) { }

    virtual ~Pass() = default;

    Result Run(ProgramModule &module);

    virtual Result PreCheck(ProgramModule &module) { return Result::Success; }

    virtual Result RunOnModule(ProgramModule &module);

    virtual Result RunOnFunction(std::shared_ptr<Function> func) = 0;

    virtual Result RunOnStatement(StatementPtr stmt) = 0;

    virtual Result RunOnOperation(OperationPtr op) = 0;

    virtual Result PostCheck(ProgramModule &module) { return  Result::Success; }

    std::string GetName() { return name_; }

private:
    std::string name_;
};

}  // namespace pto