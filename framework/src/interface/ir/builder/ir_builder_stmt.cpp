#include "ir/builder/ir_builder.h"

#include <stdexcept>
#include <utility>
#include <unordered_set>
#include <functional>

namespace pto {

OpStatement& IRBuilder::CreateOpStmt() {
    if (!compound_) throw std::runtime_error("IRBuilder::CreateOpStmt: compound is null");

    auto opStmt = std::make_shared<OpStatement>();
    auto& ref = *opStmt;
    compound_->AddStatement(std::move(opStmt));
    opStmt_ = &ref;
    return ref;
}

ForStatement& IRBuilder::CreateForStmt(std::shared_ptr<Scalar> iv, std::shared_ptr<Scalar> start, std::shared_ptr<Scalar> end, std::shared_ptr<Scalar> step) {
    if (!compound_) throw std::runtime_error("IRBuilder::CreateForStmt: compound is null");

    auto st = std::make_shared<ForStatement>(std::move(iv), std::move(start), std::move(end), std::move(step));
    auto& ref = *st;
    compound_->AddStatement(std::move(st));
    return ref;
}

IfStatement& IRBuilder::CreateIfStmt(std::string cond) {
    if (!compound_) throw std::runtime_error("IRBuilder::CreateIfStmt: compound is null");

    auto st = std::make_shared<IfStatement>(std::move(cond));
    auto& ref = *st;
    compound_->AddStatement(std::move(st));
    return ref;
}

YieldStatement& IRBuilder::CreateYield(ValuePtrs values) {
    if (!compound_) throw std::runtime_error("IRBuilder::CreateYield: compound is null");

    auto st = std::make_shared<YieldStatement>();
    st->Values() = std::move(values);

    auto& ref = *st;
    compound_->AddStatement(std::move(st));
    return ref;
}

ReturnStatement& IRBuilder::CreateReturn(ValuePtrs values) {
    if (!compound_) throw std::runtime_error("IRBuilder::CreateReturn: compound is null");

    auto st = std::make_shared<ReturnStatement>();
    st->Values() = std::move(values);

    auto& ref = *st;
    compound_->AddStatement(std::move(st));
    return ref;
}

// ===== Enter nested scopes =====
std::shared_ptr<ScopeGuard> IRBuilder::EnterFunctionBody(Function& func) {
    
    CompoundStatement& compound = func.GetCompound();
    
    return std::make_shared<ScopeGuard>(*this, &compound, &func);
}

std::shared_ptr<ScopeGuard> IRBuilder::EnterForBody(ForStatement& st) {
    if (!func_) throw std::runtime_error("IRBuilder::EnterForBody: func is null");
    if (!compound_) throw std::runtime_error("IRBuilder::EnterForBody: compound is null");

    CompoundStatement& new_compound = st.GetCompound();
    new_compound.SetParent(compound_);

    return std::make_shared<ScopeGuard>(*this, &new_compound);
}

std::shared_ptr<ScopeGuard> IRBuilder::EnterIfThen(IfStatement& st) {
    if (!func_) throw std::runtime_error("IRBuilder::EnterIfThen: func is null");
    if (!compound_) throw std::runtime_error("IRBuilder::EnterIfThen: compound is null");

    CompoundStatement& new_compound = st.GetThenCompound();
    new_compound.SetParent(compound_);

    return std::make_shared<ScopeGuard>(*this, &new_compound);
}

std::shared_ptr<ScopeGuard> IRBuilder::EnterIfElse(IfStatement& st) {
    if (!func_) throw std::runtime_error("IRBuilder::EnterIfElse: func is null");
    if (!compound_) throw std::runtime_error("IRBuilder::EnterIfElse: compound is null");

    // scope_ is already restored to the parent scope (before if) by ScopeGuard destructor
    // No need to restore environment, as scope_ already has the correct state

    CompoundStatement& new_compound = st.GetElseCompound();
    new_compound.SetParent(compound_);

    return std::make_shared<ScopeGuard>(*this, &new_compound);
}

void IRBuilder::ExitIfStatement(IfStatement& st) {
    // Get environments:
    // - envBeforeIf: environment before entering if statement (from parent scope and ancestors)
    // - envAfterThen: environment after then branch (from then scope)
    // - envAfterElse: environment after else branch (from else scope)
    CompoundStatement* parentCompound = st.GetThenCompound().GetParent();
    if (!parentCompound) {
        throw std::runtime_error("IRBuilder::ExitIfStatement: then compound has no parent");
    }
    // Get environment from parent scope, including variables from ancestor scopes
    // GetAncestorValues() returns all variables from parent scope and its ancestors
    std::unordered_map<std::string, ValuePtr> envBeforeIf = st.GetThenCompound().GetAncestorValues();
    std::unordered_map<std::string, ValuePtr> envAfterThen = st.GetThenCompound().GetEnvTable();
    std::unordered_map<std::string, ValuePtr> envAfterElse = st.GetElseCompound().GetEnvTable();

    // Check if else branch is empty
    bool hasElseBranch = !st.GetElseCompound().GetStatements().empty() || !envAfterElse.empty();
    
    // Find variables modified in any branch (by comparing with envBeforeIf)
    // This includes variables modified in both branches, or only in one branch
    std::vector<std::string> modifiedInBothBranches;
    
    if (hasElseBranch) {
        // Case 1: Has else branch - find variables modified in any branch
        // Collect all variable names that appear in either branch or existed before if
        std::unordered_set<std::string> candidateVars;
        for (const auto& [varName, _] : envAfterThen) {
            candidateVars.insert(varName);
        }
        for (const auto& [varName, _] : envAfterElse) {
            candidateVars.insert(varName);
        }
        // Also include variables that existed before if (they might be modified in one branch only)
        for (const auto& [varName, _] : envBeforeIf) {
            candidateVars.insert(varName);
        }

        // Check each candidate variable
        for (const std::string& varName : candidateVars) {
            auto itAfterThen = envAfterThen.find(varName);
            auto itAfterElse = envAfterElse.find(varName);

            bool existsInThen = (itAfterThen != envAfterThen.end());
            bool existsInElse = (itAfterElse != envAfterElse.end());
            
            // Check if variable existed before if statement by searching in parent scope chain
            ValuePtr valueBeforeIf = parentCompound->GetEnvVar(varName);
            bool existsBefore = (valueBeforeIf != nullptr);

            if (existsBefore) {
                // Variable existed before if statement
                // Check if modified in either branch
                ValuePtr valueInThen = existsInThen ? itAfterThen->second : valueBeforeIf;
                ValuePtr valueInElse = existsInElse ? itAfterElse->second : valueBeforeIf;
                bool modifiedInThen = existsInThen && (itAfterThen->second != valueBeforeIf);
                bool modifiedInElse = existsInElse && (itAfterElse->second != valueBeforeIf);
                
                if (modifiedInThen || modifiedInElse) {
                    // Modified in at least one branch - need to yield
                    modifiedInBothBranches.push_back(varName);
                }
            } else {
                // Variable didn't exist before if statement
                if (existsInThen && existsInElse) {
                    // Created in both branches - need to merge
                    modifiedInBothBranches.push_back(varName);
                }
            }
        }
    } else {
        // Case 2: No else branch - find variables modified in then branch only
        // For these variables, we'll use the original value from parent scope chain in the else branch
        for (const auto& [varName, valueAfterThen] : envAfterThen) {
            // Check if variable existed before if statement by searching in parent scope chain
            ValuePtr valueBeforeIf = parentCompound->GetEnvVar(varName);
            if (valueBeforeIf && valueBeforeIf != valueAfterThen) {
                // Modified in then branch, add to list (else branch will use original value)
                modifiedInBothBranches.push_back(varName);
            } else if (!valueBeforeIf) {
                // Variable created in then branch - for if without else, we still need to handle it
                // Use nullptr or create a default value? For now, skip variables created only in then branch
                // as they don't exist in the parent scope to merge back
            }
        }
    }

    if (modifiedInBothBranches.empty()) {
        // No variables modified in any branch, nothing to merge back into parent scope.
        return;
    }

    // Collect values to yield from each branch
    ValuePtrs thenValues;
    ValuePtrs elseValues;

    for (const std::string& varName : modifiedInBothBranches) {
        // Get original value from parent scope chain
        ValuePtr originalValue = parentCompound->GetEnvVar(varName);
        
        // Get then value: prefer from then scope if exists, otherwise use original value
        ValuePtr thenValue = nullptr;
        auto itThen = envAfterThen.find(varName);
        if (itThen != envAfterThen.end()) {
            thenValue = itThen->second;
        } else {
            // Fallback to original value if not found in then scope
            thenValue = originalValue;
        }
        
        // Get else value: prefer from else scope if exists, otherwise use original value
        ValuePtr elseValue = nullptr;
        if (hasElseBranch) {
            auto itElse = envAfterElse.find(varName);
            if (itElse != envAfterElse.end()) {
                elseValue = itElse->second;
            } else {
                // Fallback to original value if not found in else scope
                elseValue = originalValue;
            }
        } else {
            // No else branch - use original value
            elseValue = originalValue;
        }
        
        // Both values should be non-null for variables that existed before if
        if (thenValue && elseValue) {
            thenValues.push_back(thenValue);
            elseValues.push_back(elseValue);
        }
    }

    // Helper lambda to add or update yield statement in a scope
    auto addOrUpdateYield = [](std::vector<StatementPtr>& stmts, const ValuePtrs& values) {
        bool hasYield = false;
        if (!stmts.empty()) {
            hasYield = dynamic_cast<YieldStatement*>(stmts.back().get()) != nullptr;
        }
        
        if (!hasYield) {
            auto yield = std::make_shared<YieldStatement>();
            yield->Values() = values;
            stmts.push_back(yield);
        } else {
            // Update existing yield statement
            auto yield = dynamic_cast<YieldStatement*>(stmts.back().get());
            if (yield) {
                yield->Values() = values;
            }
        }
    };

    // Add or update yield statements in both branches
    addOrUpdateYield(st.GetThenCompound().GetStatements(), thenValues);
    addOrUpdateYield(st.GetElseCompound().GetStatements(), elseValues);

    // Build result using IfStatement::BuildResult()
    st.BuildResult();

    // Update parent scope environment table with merged results
    const auto& results = st.Results();
    for (size_t i = 0; i < modifiedInBothBranches.size() && i < results.size(); ++i) {
        const std::string& varName = modifiedInBothBranches[i];
        if (results[i]) {
            parentCompound->SetEnvVar(varName, results[i]);
        }
    }
}

void IRBuilder::ExitForStatement(ForStatement& st) {
    // Get environments:
    // - envBeforeFor: environment before entering for loop (from parent scope and ancestors)
    // - envAfterFor: environment after loop body (from loop scope)
    CompoundStatement* parentCompound = st.GetCompound().GetParent();
    if (!parentCompound) {
        throw std::runtime_error("IRBuilder::ExitForStatement: loop scope has no parent");
    }
    // Get environment from parent scope, including variables from ancestor scopes
    // GetAncestorValues() returns all variables from parent scope and its ancestors
    std::unordered_map<std::string, ValuePtr> envBeforeFor = st.GetCompound().GetAncestorValues();
    std::unordered_map<std::string, ValuePtr> envAfterFor = st.GetCompound().GetEnvTable();

    // Find variables that were modified in the loop body.
    // These are variables that:
    // 1. Existed before the loop (accessible from parent scope chain via GetEnvVar)
    // 2. Were modified in the loop body (different value in envAfterFor)
    std::vector<std::string> loopCarriedVars;
    
    for (const auto& [varName, valueAfterFor] : envAfterFor) {
        // Check if variable existed before loop by searching in parent scope chain
        ValuePtr valueBeforeFor = parentCompound->GetEnvVar(varName);
        if (valueBeforeFor) {
            // Variable existed before loop - check if it was modified
            if (valueBeforeFor != valueAfterFor) {
                // Variable was modified in loop body - it's a loop-carried variable
                loopCarriedVars.push_back(varName);
            }
        }
        // Note: Variables created only in the loop body are not loop-carried
        // They remain in the loop scope and don't need to be yielded
    }

    if (loopCarriedVars.empty()) {
        // No loop-carried variables, nothing to do.
        return;
    }

    // Build iterArgs from loop-carried variables
    // Use the values from envBeforeFor as initValues
    for (const std::string& varName : loopCarriedVars) {
        auto itBefore = envBeforeFor.find(varName);
        if (itBefore != envBeforeFor.end() && itBefore->second) {
            st.AddIterArg(itBefore->second);
        }
    }

    // Create value for each iter_arg and replace initValue usage in loop body
    CompoundStatement* loopCompound = &st.GetCompound();
    CompoundStatement* savedCompound = compound_;
    Function* savedFunc = func_;
    
    // Temporarily set scope and func to loop scope for creating values
    compound_ = loopCompound;
    func_ = savedFunc;  // func_ should remain the same
    
    // Helper function to create a value based on initValue type
    // Create value directly without operations (no tensor.create op)
    auto createIterArgValue = [this](ValuePtr initValue) -> ValuePtr {
        if (!initValue) return nullptr;
        
        ValueKind kind = initValue->GetValueKind();
        DataType dt = initValue->GetDataType();
        
        if (kind == ValueKind::Tensor) {
            auto tensor = std::dynamic_pointer_cast<Tensor>(initValue);
            if (tensor) {
                return CreateTensor(tensor->GetShape(), dt, tensor->GetName());
            }
        } else if (kind == ValueKind::Tile) {
            auto tile = std::dynamic_pointer_cast<Tile>(initValue);
            if (tile) {
                return CreateTile(tile->GetShape(), dt, tile->GetName());
            }
        } else if (kind == ValueKind::Scalar) {
            return CreateScalar(dt, initValue->GetName());
        }
        
        return nullptr;
    };
    
    // Create values for iter_args and store mapping from initValue to value
    std::unordered_map<ValuePtr, ValuePtr> initValueToValue;
    for (auto& iterArg : st.IterArgs()) {
        if (iterArg.initValue) {
            ValuePtr newValue = createIterArgValue(iterArg.initValue);
            if (newValue) {
                iterArg.value = newValue;
                initValueToValue[iterArg.initValue] = newValue;
            }
        }
    }
    
    // Restore scope
    compound_ = savedCompound;
    
    // Replace initValue with iter_arg value in loop body operations
    // Also update environment table
    auto replaceValueInOperations = [&initValueToValue](OpStatement& opStmt) {
        for (auto& op : opStmt.Operations()) {
            if (!op) continue;
            auto& inputs = op->MutableInputs();
            for (auto& input : inputs) {
                auto it = initValueToValue.find(input);
                if (it != initValueToValue.end()) {
                    input = it->second;
                }
            }
        }
    };
    
    // Recursive function to replace values in all nested statements
    std::function<void(StatementPtr)> replaceValueInStatement = [&](StatementPtr stmt) {
        if (!stmt) return;
        
        // Handle OpStatement
        if (auto opStmt = std::dynamic_pointer_cast<OpStatement>(stmt)) {
            replaceValueInOperations(*opStmt);
        }
        // Handle YieldStatement - replace values in yield
        else if (auto yield = std::dynamic_pointer_cast<YieldStatement>(stmt)) {
            auto& yieldValues = yield->Values();
            for (auto& val : yieldValues) {
                auto it = initValueToValue.find(val);
                if (it != initValueToValue.end()) {
                    val = it->second;
                }
            }
        }
        // Handle IfStatement - recursively process then and else branches
        else if (auto ifStmt = std::dynamic_pointer_cast<IfStatement>(stmt)) {
            // Process then branch
            for (auto& thenStmt : ifStmt->ThenBranch()) {
                replaceValueInStatement(thenStmt);
            }
            // Process else branch
            for (auto& elseStmt : ifStmt->ElseBranch()) {
                replaceValueInStatement(elseStmt);
            }
        }
        // Handle ForStatement - recursively process nested loops
        else if (auto forStmt = std::dynamic_pointer_cast<ForStatement>(stmt)) {
            for (auto& nestedStmt : forStmt->Body()) {
                replaceValueInStatement(nestedStmt);
            }
        }
    };
    
    // Replace values in all statements in loop body
    for (auto& stmt : st.GetCompound().GetStatements()) {
        replaceValueInStatement(stmt);
    }
    
    // Collect values to yield from loop body BEFORE updating environment table
    // envAfterFor contains the final computed values from the loop body (e.g., %32)
    ValuePtrs yieldValues;
    for (const std::string& varName : loopCarriedVars) {
        auto itAfter = envAfterFor.find(varName);
        if (itAfter != envAfterFor.end()) {
            // This is the final value computed in the loop body
            yieldValues.push_back(itAfter->second);
        }
    }

    // Update environment table: map variable names to iter_arg values
    // This is for the next iteration, where we use iter_arg.value as the starting point
    for (size_t i = 0; i < loopCarriedVars.size() && i < st.IterArgs().size(); ++i) {
        const std::string& varName = loopCarriedVars[i];
        auto& iterArg = st.IterArgs()[i];
        if (iterArg.value) {
            loopCompound->SetEnvVar(varName, iterArg.value);
        }
    }

    // Helper lambda to add or update yield statement in loop body
    auto addOrUpdateYield = [](std::vector<StatementPtr>& stmts, const ValuePtrs& values) {
        bool hasYield = false;
        if (!stmts.empty()) {
            hasYield = dynamic_cast<YieldStatement*>(stmts.back().get()) != nullptr;
        }
        
        if (!hasYield) {
            auto yield = std::make_shared<YieldStatement>();
            yield->Values() = values;
            stmts.push_back(yield);
        } else {
            // Update existing yield statement
            auto yield = dynamic_cast<YieldStatement*>(stmts.back().get());
            if (yield) {
                yield->Values() = values;
            }
        }
    };

    // Add or update yield statement in loop body
    addOrUpdateYield(st.GetCompound().GetStatements(), yieldValues);

    // Build result using ForStatement::BuildResult()
    st.BuildResult();

    // Update parent scope environment table with loop results
    const auto& results = st.Results();
    for (size_t i = 0; i < loopCarriedVars.size() && i < results.size(); ++i) {
        const std::string& varName = loopCarriedVars[i];
        if (results[i]) {
            parentCompound->SetEnvVar(varName, results[i]);
        }
    }
}

} // namespace pto
