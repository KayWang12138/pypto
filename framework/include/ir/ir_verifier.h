// ir/ir_verifier.h
// PTO-IR prototype: IRVerifier (basic structural + opcode/operand sanity).
// All comments must remain in English for consistency.

#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "ir/utils_defop_switch.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/operation.h"
#include "ir/opcode.h"

namespace pto {

class IRVerifier {
public:
    struct Result {
        bool ok{true};
        std::vector<std::string> errors;

        void Add(std::string msg) { errors.push_back(std::move(msg)); ok = false; }

        std::string ToString() const {
            std::ostringstream oss;
            oss << (ok ? "IRVerifier: OK\n" : "IRVerifier: FAILED\n");
            for (auto& e : errors) oss << "  - " << e << "\n";
            return oss.str();
        }
    };

    Result VerifyFunction(const std::shared_ptr<Function>& f) {
        Result r;
        if (!f) { r.Add("Function is null"); return r; }
        if (!f->GetCompound()) { r.Add("Function body is null"); return r; }
        VerifyCompound(r, *f->GetCompound(), "func");
        if (r.errors.empty()) r.ok = true;
        return r;
    }

    // ===== operation dispatch =====
    void Verify(Operation& op, Result& r, const std::string& path) {
        switch (op.GetOpcode()) {

#define PTO_VERIFIER_HANDLE(OPCLASS, OPC) \
    case Opcode::OPC: \
        this->Verify##OPCLASS(static_cast<OPCLASS&>(op), r, path); \
        break;

#define DEFOP(OPCLASS, inherit_token, opcode_token, ...) \
    PTO_DEFOP_SWITCH(OPCLASS, opcode_token, PTO_VERIFIER_HANDLE)

#include "ir/operation.def"
#include "ir/tile_graph.def"

#undef DEFOP
#undef PTO_VERIFIER_HANDLE

        default:
            VerifyOperation(op, r, path);
            break;
        }
    }

    // ===== default per-opclass handlers =====
#define PTO_DECLARE_DEFAULT_VERIFY(OPCLASS, inherit_token, opcode_token, ...) \
    void Verify##OPCLASS(OPCLASS& op, Result& r, const std::string& path) { \
        VerifyOperation(op, r, path); \
    }

#define DEFOP(OPCLASS, inherit_token, opcode_token, ...) \
    PTO_DECLARE_DEFAULT_VERIFY(OPCLASS, inherit_token, opcode_token, __VA_ARGS__)

#include "ir/operation.def"
#include "ir/tile_graph.def"

#undef DEFOP
#undef PTO_DECLARE_DEFAULT_VERIFY

    // ===== generic operation verifier =====
    void VerifyOperation(Operation& op, Result& r, const std::string& path) {
        if (op.GetOpcode() == Opcode::OP_INVALID) {
            r.Add(path + ": opcode is OP_INVALID");
        }

        for (size_t i = 0; i < op.GetNumInputOperand(); ++i) {
            if (!op.GetInputOperand(i)) r.Add(path + ": input[" + std::to_string(i) + "] is null");
        }
        for (size_t i = 0; i < op.GetNumOutputOperand(); ++i) {
            if (!op.GetOutputOperand(i)) r.Add(path + ": output[" + std::to_string(i) + "] is null");
        }

        if (op.iScalarIndex_ < -1) r.Add(path + ": iScalarIndex_ < -1");
        if (op.oScalarIndex_ < -1) r.Add(path + ": oScalarIndex_ < -1");

        if (op.iScalarIndex_ >= 0 && (size_t)op.iScalarIndex_ > op.ioperands_.size())
            r.Add(path + ": iScalarIndex_ out of range");
        if (op.oScalarIndex_ >= 0 && (size_t)op.oScalarIndex_ > op.ooperands_.size())
            r.Add(path + ": oScalarIndex_ out of range");
    }

private:
    // ===== statements =====
    void VerifyCompound(Result& r, CompoundStatement& cs, const std::string& path) {
        auto stmts = cs.GetStatements(); // your const returns by value; OK for verify.
        for (size_t i = 0; i < stmts.size(); ++i) {
            if (!stmts[i]) { r.Add(path + ": stmt[" + std::to_string(i) + "] is null"); continue; }
            VerifyStatement(r, *stmts[i], path + "/stmt[" + std::to_string(i) + "]");
        }
    }

    void VerifyStatement(Result& r, Statement& st, const std::string& path) {
        switch (st.GetKind()) {
        case StatementKind::Compound:
            VerifyCompound(r, static_cast<CompoundStatement&>(st), path);
            break;
        case StatementKind::Op:
            VerifyOpStatement(r, static_cast<OpStatement&>(st), path);
            break;
        case StatementKind::For:
            VerifyFor(r, static_cast<ForStatement&>(st), path);
            break;
        case StatementKind::If:
            VerifyIf(r, static_cast<IfStatement&>(st), path);
            break;
        case StatementKind::Yield:
            VerifyYield(r, static_cast<YieldStatement&>(st), path);
            break;
        case StatementKind::Return:
            VerifyReturn(r, static_cast<ReturnStatement&>(st), path);
            break;
        default:
            r.Add(path + ": unknown statement kind");
            break;
        }
    }

    void VerifyOpStatement(Result& r, OpStatement& os, const std::string& path) {
        const auto& ops = os.Operations();
        for (size_t i = 0; i < ops.size(); ++i) {
            if (!ops[i]) { r.Add(path + ": op[" + std::to_string(i) + "] is null"); continue; }
            Verify(*ops[i], r, path + "/op[" + std::to_string(i) + "]");
        }
    }

    void VerifyYield(Result& r, YieldStatement& ys, const std::string& path) {
        const auto& vals = ys.Values();
        if (vals.empty()) r.Add(path + ": yield has empty values");
        for (size_t i = 0; i < vals.size(); ++i) {
            if (!vals[i]) r.Add(path + ": yield value[" + std::to_string(i) + "] is null");
        }
    }

    void VerifyReturn(Result& r, ReturnStatement& rs, const std::string& path) {
        const auto& vals = rs.Values();
        for (size_t i = 0; i < vals.size(); ++i) {
            if (!vals[i]) r.Add(path + ": return value[" + std::to_string(i) + "] is null");
        }
    }

    void VerifyIf(Result& r, IfStatement& is, const std::string& path) {
        if (!is.GetThenCompound()) r.Add(path + ": then compound is null");
        if (!is.GetElseCompound()) r.Add(path + ": else compound is null");

        if (is.GetThenCompound()) VerifyCompound(r, *is.GetThenCompound(), path + "/then");
        if (is.GetElseCompound()) VerifyCompound(r, *is.GetElseCompound(), path + "/else");

        auto thenStmts = is.ThenBranch();
        auto elseStmts = is.ElseBranch();
        auto* thenYield = (!thenStmts.empty()) ? dynamic_cast<YieldStatement*>(thenStmts.back().get()) : nullptr;
        auto* elseYield = (!elseStmts.empty()) ? dynamic_cast<YieldStatement*>(elseStmts.back().get()) : nullptr;

        if (!thenYield || !elseYield) {
            r.Add(path + ": if branches must end with YieldStatement");
            return;
        }
        if (thenYield->Values().size() != elseYield->Values().size()) {
            r.Add(path + ": then/else yield value count mismatch");
        }
        if (!is.Results().empty() && is.Results().size() != thenYield->Values().size()) {
            r.Add(path + ": if results size != yield size (did you call BuildResult?)");
        }
    }

    void VerifyFor(Result& r, ForStatement& fs, const std::string& path) {
        if (!fs.GetCompound()) r.Add(path + ": for body compound is null");
        if (fs.GetCompound()) VerifyCompound(r, *fs.GetCompound(), path + "/body");

        if (auto y = fs.Yield()) {
            VerifyYield(r, *y, path + "/yield");
            if (!fs.Results().empty() && fs.Results().size() != y->Values().size()) {
                r.Add(path + ": for results size != yield size (did you call BuildResult?)");
            }
        }
        // If your semantics require loop to always yield, turn missing yield into error here.
    }
};

} // namespace pto
