#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include "ir/function.h"
#include "ir/statement.h"

namespace pto {

// A single stack frame for restoring insertion state.
struct ScopeFrame {
    std::shared_ptr<CompoundStatement> compound{nullptr};
    std::shared_ptr<Function> func{nullptr};
    std::shared_ptr<OpStatement> activeOpStmt{nullptr};
};

// Explicit, stack-based builder context.
// All mutable insertion state lives here.
struct IRBuilderContext {
    std::shared_ptr<Function> func{nullptr};
    std::shared_ptr<CompoundStatement> compound{nullptr};
    std::shared_ptr<OpStatement> activeOpStmt{nullptr};

    std::vector<ScopeFrame> scopeStack;

    void ResetInsertionPoint() { activeOpStmt.reset(); }

    void PushScope(std::shared_ptr<CompoundStatement> newCompound,
                   std::shared_ptr<Function> newFunc = nullptr) {
        scopeStack.push_back(ScopeFrame{compound, func, activeOpStmt});

        compound = std::move(newCompound);
        if (newFunc) {
            func = std::move(newFunc);
        }
        activeOpStmt.reset();  // always reset insertion point when entering a new scope
    }

    void PopScope() {
        assert(!scopeStack.empty() && "IRBuilderContext::PopScope: scopeStack is empty");

        ScopeFrame frame = scopeStack.back();
        scopeStack.pop_back();

        compound = std::move(frame.compound);
        func = std::move(frame.func);
        activeOpStmt = std::move(frame.activeOpStmt);
    }
};

} // namespace pto
