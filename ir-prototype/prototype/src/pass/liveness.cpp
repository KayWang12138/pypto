// liveness.cpp
#include "pass/liveness.h"
#include <iostream>

namespace pto {

// ------------------------------------------------------------
// Collect DEF set of a Statement
// ------------------------------------------------------------
ValueSet LivenessAnalyzer::CollectDef(Statement* stmt) {
    ValueSet def;

    switch (stmt->GetKind()) {
    case StatementKind::Block: {
        auto* block = static_cast<BlockStatement*>(stmt);
        for (auto& op : block->Operations()) {
            for (auto& out : op->GetOutputs())
                def.insert(out);
        }
        break;
    }
    case StatementKind::Call: {
        auto* call = static_cast<CallStatement*>(stmt);
        for (auto& r : call->Results())
            def.insert(r);
        break;
    }
    case StatementKind::If: {
        auto* ifs = static_cast<IfStatement*>(stmt);
        for (auto& r : ifs->Results())
            def.insert(r);
        break;
    }
    case StatementKind::For: {
        auto* fs = static_cast<ForStatement*>(stmt);
        // TODO: iterVar + iterArgs + result
        for (auto& r : fs->Results()) {
            def.insert(r);
        }
        break;
    }
    // Yield
    default:
        break;
    }

    stmtDef_.insert({stmt, def});
    return def;
}

// ------------------------------------------------------------
// Collect USE set of a Statement
// ------------------------------------------------------------
ValueSet LivenessAnalyzer::CollectUse(Statement* stmt) {
    if (stmtUse_.find(stmt) != stmtUse_.end()) {
        return stmtUse_.at(stmt);
    }

    ValueSet use;

    switch (stmt->GetKind()) {
    case StatementKind::Block: {
        auto* block = static_cast<BlockStatement*>(stmt);
        for (auto& op : block->Operations()) {
            for (auto& in : op->GetInputs())
                use.insert(in);
        }
        break;
    }
    case StatementKind::Call: {
        auto* call = static_cast<CallStatement*>(stmt);
        for (auto& a : call->Arguments())
            use.insert(a);
        break;
    }
    case StatementKind::If: {
        auto* ifs = static_cast<IfStatement*>(stmt);
        // condittion use + thenScope use + elseScope use
        // TODO: condition use

        // thenScope use
        auto& ts = ifs->GetThenScope();
        for (auto s : ts.GetStatements()) {
            auto thenUse = CollectUse(s.get());
            use.merge(thenUse);
        }

        // elseScope use
        auto& es = ifs->GetElseScope();
        for (auto s : es.GetStatements()) {
            auto elseUse = CollectUse(s.get());
            use.merge(elseUse);
        }
        break;
    }
    case StatementKind::For: {
        auto *fs = static_cast<ForStatement*>(stmt);
        // TODO: lb + ub + step + iter_init + body
        // now only body is Value
        for (auto s : fs->Body()) {
            auto bodyUse = CollectUse(s.get());
            use.merge(bodyUse);
        }
        break;
    }
    case StatementKind::Yield: {
        auto* ys = static_cast<YieldStatement*>(stmt);
        for (auto& v : ys->Values())
            use.insert(v);
        break;
    }
    case StatementKind::Return: {
        auto* rs = static_cast<ReturnStatement*>(stmt);
        for (auto& v : rs->Values())
            use.insert(v);
        break;
    }
    default:
        break;
    }

    stmtUse_.insert({stmt, use});
    return use;
}

// ------------------------------------------------------------
// Compute OUT[S] given linear order + structured control flow
// ------------------------------------------------------------
ValueSet LivenessAnalyzer::ComputeOut(
    Statement* stmt,
    const std::vector<StatementPtr>& stmts,
    size_t index)
{
    ValueSet out;

    // control flow scope analyze alone.
    switch (stmt->GetKind()) {
    case StatementKind::If: {
        auto* ifs = static_cast<IfStatement*>(stmt);
        if (!ifs->ThenBranch().empty()) {
            AnalyzeStatementList(ifs->ThenBranch());
        }
        if (!ifs->ElseBranch().empty())
            AnalyzeStatementList(ifs->ElseBranch());
        break;
    }
    case StatementKind::For: {
        auto* fs = static_cast<ForStatement*>(stmt);
        if (!fs->Body().empty()) {
            AnalyzeStatementList(fs->Body());
        }
        break;
    }
    case StatementKind::Return:
        // no successors
        return out;
    default:
        break;
    }
    // control flow is also a statement, computed like block. 
    // Each statement has only one successor in same scope.
    if (index + 1 < stmts.size()) {
        auto& suIn = gLiveness_[stmts[index + 1].get()].in;
        out.insert(suIn.begin(), suIn.end());
    }
    return out;
}

// ------------------------------------
// Analyze a linear list of statements 
// ------------------------------------
void LivenessAnalyzer::AnalyzeStatementList(const std::vector<StatementPtr>& stmts) {
    // Initialize
    for (auto& s : stmts) {
        gLiveness_[s.get()] = LivenessInfo{};
    }

    bool changed = true;
    while (changed) {
        changed = false;

        for (int i = static_cast<int>(stmts.size()) - 1; i >= 0; --i) {
            Statement* S = stmts[i].get();

            auto oldIn  = gLiveness_[S].in;
            auto oldOut = gLiveness_[S].out;

            // OUT[S] = U(IN[S_successor])
            gLiveness_[S].out = ComputeOut(S, stmts, i);

            ValueSet in = CollectUse(S);
            ValueSet def = CollectDef(S);

#ifdef DEBUG       
            std::cout << "use (";
            for (auto inn : in) {
                std::cout << inn->GetSSAName() << ", ";
            }
            std::cout << ")" << std::endl << "def (";
            for (auto onn : def) {
                std::cout << onn->GetSSAName() << ", ";
            }
            std::cout << ")" << std::endl;
#endif

            for (auto& v : gLiveness_[S].out)
                if (!def.count(v))
                    in.insert(v);

            // IN[S] = use[S] + (OUT[S] - def[S])
            gLiveness_[S].in = std::move(in);

            if (gLiveness_[S].in != oldIn ||
                gLiveness_[S].out != oldOut) {
                changed = true;
            }
#ifdef DEBUG                
            std::cout << "in (";
            for (auto inn : gLiveness_[S].in) {
                std::cout << inn->GetSSAName() << ", ";
            }
            std::cout << ")" << std::endl << "out (";
            for (auto onn : gLiveness_[S].out) {
                std::cout << onn->GetSSAName() << ", ";
            }
            std::cout << ")" << std::endl;
#endif
        }
    }
}

// ------------------------------------------------------------
// Public entry: run liveness on a Function
// ------------------------------------------------------------
void LivenessAnalyzer::RunLivenessAnalysis(Function& func) {
    gLiveness_.clear();
    stmtUse_.clear();
    stmtDef_.clear();
    AnalyzeStatementList(func.Body());
}

// ------------------------------------------------------------
// Query API
// ------------------------------------------------------------
const ValueSet& LivenessAnalyzer::GetLiveIn(StatementPtr stmt) {
    return gLiveness_.at(stmt.get()).in;
}

const ValueSet& LivenessAnalyzer::GetLiveOut(StatementPtr stmt) {
    return gLiveness_.at(stmt.get()).out;
}

const ValueSet& LivenessAnalyzer::GetUse(StatementPtr stmt) {
    return stmtUse_.at(stmt.get());
}

const ValueSet& LivenessAnalyzer::GetDef(StatementPtr stmt) {
    return stmtDef_.at(stmt.get());
}

} // namespace pto
