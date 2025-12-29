#pragma once

#include "pass/pass.h"
#include "ir/program.h"

#include <memory>
#include <vector>

namespace pto {

class PassManager {
public:
    static PassManager& Instance() {
        static PassManager pm;
        return pm;
    }

    PassManager(const PassManager&) = delete;
    PassManager& operator=(const PassManager&) = delete;
    ~PassManager() = default;

    void AddPass(std::shared_ptr<Pass> pass) {
        passes_.push_back(pass);
    }

    void Clear() { passes_.clear(); }

    Result Run(ProgramModule &module, bool print = false);
    
    Result Run(std::shared_ptr<Function> func, bool print = false);

private:
    PassManager();

    std::vector<std::shared_ptr<Pass>> passes_;
};

}  // namespace pto

