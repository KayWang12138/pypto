#include "pass/dump_ir_pass.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/operation.h"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

namespace pto {

DumpIRPass::DumpIRPass(const std::string& name)
    : Pass(name), output_dir_(""), dump_to_terminal_(true) {
}

DumpIRPass::DumpIRPass(const std::string& name, const std::string& output_dir)
    : Pass(name), output_dir_(output_dir), dump_to_terminal_(output_dir.empty()) {
    if (!dump_to_terminal_) {
        EnsureDirectoryExists(output_dir_);
    }
}

void DumpIRPass::SetOutputDir(const std::string& output_dir) {
    output_dir_ = output_dir;
    dump_to_terminal_ = output_dir.empty();
    if (!dump_to_terminal_) {
        EnsureDirectoryExists(output_dir_);
    }
}

bool DumpIRPass::EnsureDirectoryExists(const std::string& dir) {
    struct stat info;
    
    // Check if directory exists
    if (stat(dir.c_str(), &info) != 0) {
        // Directory doesn't exist, try to create it
        if (mkdir(dir.c_str(), 0755) != 0) {
            std::cerr << "Error: Failed to create directory '" << dir << "'" << std::endl;
            return false;
        }
        std::cout << "Created directory: " << dir << std::endl;
        return true;
    } else if (info.st_mode & S_IFDIR) {
        // Directory exists
        return true;
    } else {
        std::cerr << "Error: '" << dir << "' exists but is not a directory" << std::endl;
        return false;
    }
}

std::string DumpIRPass::GetOutputFilePath(const std::string& func_name) {
    // Create file path: output_dir/function_name.ir
    // func_name should be the raw name without prefix
    std::string file_path = output_dir_;
    if (!file_path.empty() && file_path.back() != '/') {
        file_path += '/';
    }
    file_path += func_name + ".ir";
    return file_path;
}

void DumpIRPass::DumpFunctionToTerminal(std::shared_ptr<Function> func) {
    std::cout << *func << std::endl;
}

void DumpIRPass::DumpFunctionToFile(std::shared_ptr<Function> func) {
    // Use GetName() to get the raw name without @ prefix for filename
    std::string file_path = GetOutputFilePath(func->GetName());
    std::ofstream file_stream(file_path, std::ios::out | std::ios::trunc);
    
    if (!file_stream.is_open()) {
        std::cerr << "Error: Failed to open file '" << file_path 
                  << "' for dumping IR" << std::endl;
        return;
    }
    
    // Print uses GetPrefixedName() which includes @ prefix
    file_stream << *func << std::endl;
    file_stream.close();
    
    std::cout << "Dumped function '" << func->GetPrefixedName() 
              << "' to: " << file_path << std::endl;
}

void DumpIRPass::DumpModuleToTerminal(const std::string& name, ProgramModule& module) {
    std::cout << "============= after pass: " << name
              << " ===============" << std::endl;
    std::cout << module << std::endl << std::endl;
}

void DumpIRPass::DumpModuleToFile(const std::string& name, ProgramModule& module) {
    // Use GetName() to get the raw name without @ prefix for filename
    std::string file_path = GetOutputFilePath(module.GetProgramEntry()->GetName());
    std::ofstream file_stream(file_path, std::ios::out | std::ios::app);
    
    if (!file_stream.is_open()) {
        std::cerr << "Error: Failed to open file '" << file_path 
                  << "' for dumping IR" << std::endl;
        return;
    }
    
    file_stream << "============= after pass: " << name 
                << " ===============" << std::endl;
    file_stream << module << std::endl << std::endl;
    file_stream.close();
}

Result DumpIRPass::RunOnModule(ProgramModule& module) {
    if (dump_to_terminal_) {
        DumpModuleToTerminal(GetName(), module);
    } else {
        std::string file = GetOutputFilePath(module.GetProgramEntry()->GetName());
        if (std::filesystem::exists(file)) {
            std::filesystem::remove(file);
        }
        DumpModuleToFile(GetName(), module);
    }
    return Result::Success;
}

Result DumpIRPass::RunOnFunction(std::shared_ptr<Function> func) {
    if (!func) {
        return Result::Failure;
    }
    
    if (dump_to_terminal_) {
        DumpFunctionToTerminal(func);
    } else {
        DumpFunctionToFile(func);
    }
    
    return Result::Success;
}

Result DumpIRPass::RunOnStatement(StatementPtr stmt) {
    // Statements are handled by function printing
    return Result::Success;
}

Result DumpIRPass::RunOnOperation(OperationPtr op) {
    // Operations are handled by function printing
    return Result::Success;
}

void DumpIRPass::DumpIRAfterPass(const std::string& name, ProgramModule& module) {
    if (dump_to_terminal_) {
        DumpModuleToTerminal(name, module);
    } else {
        DumpModuleToFile(name, module);
    }
}

}  // namespace pto

