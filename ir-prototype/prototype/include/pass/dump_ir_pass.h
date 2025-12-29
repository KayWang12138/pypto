#pragma once

#include "ir/program.h"
#include "pass/pass.h"

#include <fstream>
#include <memory>
#include <string>

namespace pto {

// Pass for dumping IR to terminal or directory (creates function_name.ir files)
class DumpIRPass : public Pass {
public:
    // Constructor for dumping to terminal
    explicit DumpIRPass(const std::string& name = "DumpIR");
    
    // Constructor for dumping to directory (creates function_name.ir files)
    DumpIRPass(const std::string& name, const std::string& output_dir);
    
    ~DumpIRPass() override = default;

    Result RunOnModule(ProgramModule& module) override;

    Result RunOnFunction(std::shared_ptr<Function> func) override;
    
    Result RunOnStatement(StatementPtr stmt) override;
    
    Result RunOnOperation(OperationPtr op) override;
    
    // Set output directory path
    void SetOutputDir(const std::string& output_dir);
    
    // Get output directory path
    std::string GetOutputDir() const { return output_dir_; }
    
    // Check if dumping to directory
    bool IsDumpingToDir() const { return !output_dir_.empty(); }

    void DumpIRAfterPass(const std::string& name, ProgramModule& module);

private:
    std::string output_dir_;
    bool dump_to_terminal_;
    
    void DumpModuleToTerminal(const std::string& name, ProgramModule& module);
    void DumpModuleToFile(const std::string& name, ProgramModule& module);
    void DumpFunctionToTerminal(std::shared_ptr<Function> func);
    void DumpFunctionToFile(std::shared_ptr<Function> func);
    std::string GetOutputFilePath(const std::string& func_name);
    bool EnsureDirectoryExists(const std::string& dir);
};

}  // namespace pto

