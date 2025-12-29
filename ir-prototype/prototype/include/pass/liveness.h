#pragma once

#include "ir/function.h"
#include "ir/statement.h"
#include "ir/type.h"

#include <unordered_map>
#include <unordered_set>

namespace pto {

using ValueSet = std::unordered_set<ValuePtr>;

// ------------------------------------------------------------
// Liveness information attached to each Statement
// ------------------------------------------------------------
struct LivenessInfo {
    ValueSet in;
    ValueSet out;
};

class LivenessAnalyzer {
public:
    // Public entry: run liveness on a Function
    void RunLivenessAnalysis(Function& func);

    // Query API
    const ValueSet& GetLiveIn(StatementPtr stmt);
    const ValueSet& GetLiveOut(StatementPtr stmt);

    const ValueSet& GetUse(StatementPtr stmt);
    const ValueSet& GetDef(StatementPtr stmt);

private:
    // Collect DEF/USE set of a Statement
    ValueSet CollectDef(Statement* stmt);
    ValueSet CollectUse(Statement* stmt);

    // Analyze a linear list of statements 
    void AnalyzeStatementList(const std::vector<StatementPtr>& stmts);

    // Compute OUT[S] given linear order + structured control flow
    ValueSet ComputeOut(Statement* stmt, const std::vector<StatementPtr>& stmts, size_t index);

    std::unordered_map<Statement*, LivenessInfo> gLiveness_;
    std::unordered_map<Statement*, ValueSet> stmtUse_;
    std::unordered_map<Statement*, ValueSet> stmtDef_;
};

}