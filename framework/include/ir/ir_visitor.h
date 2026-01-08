// ir/ir_visitor.h
// PTO-IR prototype: IRVisitor (CRTP) for Operation/Statement traversal.
// All comments must remain in English for consistency.

#pragma once

#include "ir/utils_defop_switch.h"
#include "ir/opcode.h"
#include "ir/operation.h"
#include "ir/statement.h"
#include "ir/function.h"
#include "ir/program.h"

namespace pto {

template <class Derived, class RetTy = void>
class IRVisitor {
public:
    using RetType = RetTy;

    // ====== Operation dispatch ======
    RetTy Visit(Operation& op) {
        switch (op.GetOpcode()) {

#define PTO_VISITOR_HANDLE(OPCLASS, OPC) \
    case Opcode::OPC: \
        return static_cast<Derived*>(this)->Visit##OPCLASS(static_cast<OPCLASS&>(op));

#define DEFOP(OPCLASS, inherit_token, opcode_token, ...) \
    PTO_DEFOP_SWITCH(OPCLASS, opcode_token, PTO_VISITOR_HANDLE)
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP
#undef PTO_VISITOR_HANDLE

        default:
            return static_cast<Derived*>(this)->VisitOperation(op);
        }
    }

    // ====== Statement dispatch ======
    RetTy VisitStatement(Statement& st) {
        switch (st.GetKind()) {
        case StatementKind::Compound:
            return static_cast<Derived*>(this)->VisitCompoundStatement(static_cast<CompoundStatement&>(st));
        case StatementKind::Op:
            return static_cast<Derived*>(this)->VisitOpStatement(static_cast<OpStatement&>(st));
        case StatementKind::For:
            return static_cast<Derived*>(this)->VisitForStatement(static_cast<ForStatement&>(st));
        case StatementKind::If:
            return static_cast<Derived*>(this)->VisitIfStatement(static_cast<IfStatement&>(st));
        case StatementKind::Yield:
            return static_cast<Derived*>(this)->VisitYieldStatement(static_cast<YieldStatement&>(st));
        case StatementKind::Return:
            return static_cast<Derived*>(this)->VisitReturnStatement(static_cast<ReturnStatement&>(st));
        default:
            return static_cast<Derived*>(this)->VisitUnknownStatement(st);
        }
    }

    // ====== Default handlers ======
    RetTy VisitOperation(Operation&) { return RetTy(); }

#define PTO_DECLARE_DEFAULT_VISIT(OPCLASS, inherit_token, opcode_token, ...) \
    RetTy Visit##OPCLASS(OPCLASS& op) { \
        return static_cast<Derived*>(this)->VisitOperation(op); \
    }

#define DEFOP(OPCLASS, inherit_token, opcode_token, ...) \
    PTO_DECLARE_DEFAULT_VISIT(OPCLASS, inherit_token, opcode_token, __VA_ARGS__)

#include "ir/operation.def"
#include "ir/tile_graph.def"

#undef DEFOP
#undef PTO_DECLARE_DEFAULT_VISIT

    // Statement defaults
    RetTy VisitUnknownStatement(Statement&) { return RetTy(); }

    RetTy VisitCompoundStatement(CompoundStatement& cs) {
        for (size_t i = 0; i < cs.GetStatementsNum(); ++i) {
            auto st = cs.GetStatement(i);
            if (st) static_cast<Derived*>(this)->VisitStatement(*st);
        }
        return RetTy();
    }

    RetTy VisitOpStatement(OpStatement& os) {
        for (auto& op : os.Operations()) {
            if (op) static_cast<Derived*>(this)->Visit(*op);
        }
        return RetTy();
    }

    RetTy VisitForStatement(ForStatement& fs) {
        if (fs.GetCompound()) static_cast<Derived*>(this)->VisitCompoundStatement(*fs.GetCompound());
        return RetTy();
    }

    RetTy VisitIfStatement(IfStatement& is) {
        if (is.GetThenCompound()) static_cast<Derived*>(this)->VisitCompoundStatement(*is.GetThenCompound());
        if (is.GetElseCompound()) static_cast<Derived*>(this)->VisitCompoundStatement(*is.GetElseCompound());
        return RetTy();
    }

    RetTy VisitYieldStatement(YieldStatement&) { return RetTy(); }
    RetTy VisitReturnStatement(ReturnStatement&) { return RetTy(); }

    // ====== Function and Program dispatch ======
    RetTy VisitFunction(Function& func) {
        if (auto compound = func.GetCompound()) {
            static_cast<Derived*>(this)->VisitCompoundStatement(*compound);
        }
        return RetTy();
    }

    RetTy VisitProgramModule(ProgramModule& pm) {
        // Visit program entry if it exists
        if (auto entry = pm.GetProgramEntry()) {
            static_cast<Derived*>(this)->VisitFunction(*entry);
        }
        // Visit all functions
        for (const auto& func : pm.GetFunctions()) {
            if (func) static_cast<Derived*>(this)->VisitFunction(*func);
        }
        return RetTy();
    }
};

} // namespace pto
