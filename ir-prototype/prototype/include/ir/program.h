// PTO-IR prototype: program-level IR structures.
// All comments must remain in English for consistency.

#pragma once

#include "ir/utils.h"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace pto {

class Function;

// Describes a program entry point, which references a function symbol.
struct ProgramEntry {
    std::string funcName; // e.g. "@main_function"
};

// Represents the top-level program.module container.
class ProgramModule : public Object, public AttributeHolder {
public:
    explicit ProgramModule(std::string name);

    ObjectType GetObjectType() const override { return ObjectType::Program; }

    // Entrypoints.
    void SetProgramEntry(const std::shared_ptr<Function>& programEntry);
    const std::shared_ptr<Function> GetProgramEntry() const { return programEntry_; }
    
    // Functions (defined in func.h).
    void AddFunction(const std::shared_ptr<Function>& function);
    const std::vector<std::shared_ptr<Function>> GetFunctions() const { return functions_; }

    // Pretty-print to a textual PTO-IR-like form.
    void Print(std::ostream& os, int indent = 0) const;

private:
    std::shared_ptr<Function> programEntry_;
    std::vector<std::shared_ptr<Function>> functions_;
};

// Helper for convenient streaming: std::cout << module;
std::ostream& operator<<(std::ostream& os, const ProgramModule& module);

} // namespace pto


