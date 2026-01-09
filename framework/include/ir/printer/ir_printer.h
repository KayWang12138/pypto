// ir/ir_printer.h
#pragma once

#include <ostream>

#include "ir/ir_visitor.h"
#include "ir/object.h"
#include "ir/printer/ir_opcode_formatter.h"

namespace pto {

class IRPrinter : public IRVisitor<IRPrinter> {
public:
    explicit IRPrinter(std::ostream& os)
        : os_(os), formatter_(os) {}

    // ===== entry =====
    void VisitProgramModule(ProgramModule& pm) {
        // Print module header
        PrintIndent();
        os_ << "program.module " << pm.GetPrefixedName() << " {\n";

        // Print program entry
        Indent();
        PrintIndent();
        if (auto entry = pm.GetProgramEntry()) {
            os_ << "program.entry " << entry->GetPrefixedName() << "\n";
        }

        // Print program-level attributes
        const auto& attrs = pm.Attributes();
        if (!attrs.empty()) {
            for (const auto& kv : attrs) {
                PrintIndent();
                os_ << "attr " << kv.first << " = " << kv.second << "\n";
            }
        }

        // Visit all functions using visitor pattern
        for (const auto& func : pm.GetFunctions()) {
            if (func) {
                VisitFunction(*func);
                os_ << "\n";
            }
        }
        Dedent();

        // Closing brace
        PrintIndent();
        os_ << "}\n";
    }

    // ===== function =====
    void VisitFunction(Function& func) {
        // Print function header
        PrintIndent();
        os_ << "func.func " << func.GetPrefixedName() << "(";
        const auto& signature = func.GetSignature();
        for (size_t i = 0; i < signature.arguments.size(); ++i) {
            const auto& arg = signature.arguments[i];
            if (arg) {
                arg->PrintValue(os_);
            }
            if (i + 1 < signature.arguments.size()) {
                os_ << ", ";
            }
        }
        os_ << ")";

        if (!signature.results.empty()) {
            os_ << " -> (";
            for (size_t i = 0; i < signature.results.size(); ++i) {
                if (signature.results[i]) {
                    signature.results[i]->PrintType(os_);
                }
                if (i + 1 < signature.results.size()) {
                    os_ << ", ";
                }
            }
            os_ << ")";
        }
        os_ << " {\n";

        // Use Visitor to print function body statements
        Indent();
        if (auto compound = func.GetCompound()) {
            VisitCompoundStatement(*compound);
        }
        Dedent();

        // Closing brace
        PrintIndent();
        os_ << "}\n";

        // Print function attributes
        std::string attrPrefix = "func.attr ";
        PrintAttributes(func.Attributes(), attrPrefix);

        // Print function kind comment
        PrintIndent();
        os_ << "// func.kind = " << ToString(func.GetKind()) << "\n";
    }

    // ===== statements =====

    void VisitCompoundStatement(CompoundStatement& cs) {
        for (size_t i = 0; i < cs.GetStatementsNum(); ++i) {
            auto st = cs.GetStatement(i);
            if (st) VisitStatement(*st);
        }
    }

    void VisitOpStatement(OpStatement& os) {
        PrintIndent();
        os_ << "statement.op {\n";

        // Print linear operations.
        Indent();
        const auto& operations = os.Operations();
        for (const auto& op : operations) {
            if (op) {
                PrintIndent();
                formatter_.Print(*op);
            }
        }
        Dedent();
        PrintIndent();
        os_ << "}\n";
    }

    void VisitIfStatement(IfStatement& is) {
        PrintIndent();
        // If results_ is non-empty, print them as the result variables:
        //   %r0, %r1 = statement.if ...
        // Otherwise, treat this as a pure statement-level conditional.
        const auto& results = is.Results();
        if (!results.empty()) {
            for (size_t i = 0; i < results.size(); ++i) {
                if (results[i]) {
                    results[i]->PrintSSAName(os_);
                }
                if (i + 1 < results.size()) {
                    os_ << ", ";
                }
            }
            os_ << " = ";
        }

        os_ << "statement.if ";
        if (auto cond = is.GetCondition()) {
            cond->PrintSSAName(os_);
        }
        os_ << " {\n";

        Indent();
        if (auto thenCompound = is.GetThenCompound()) {
            VisitCompoundStatement(*thenCompound);
        }
        Dedent();

        PrintIndent();
        os_ << "} else {\n";

        Indent();
        if (auto elseCompound = is.GetElseCompound()) {
            VisitCompoundStatement(*elseCompound);
        }
        Dedent();
        
        PrintIndent();
        os_ << "}\n";
    }

    void VisitForStatement(ForStatement& fs) {
        PrintIndent();

        // Print result variables if results_ is non-empty
        const auto& results = fs.Results();
        if (!results.empty()) {
            for (size_t i = 0; i < results.size(); ++i) {
                if (results[i]) {
                    results[i]->PrintSSAName(os_);
                }
                if (i + 1 < results.size()) {
                    os_ << ", ";
                }
            }
            os_ << " = ";
        }

        // Print loop header: statement.for %iv = %lb to %ub step %step
        os_ << "statement.for ";
        if (auto iv = fs.GetIterationVar()) {
            iv->PrintSSAName(os_);
        } else {
            os_ << "<null>";
        }
        os_ << " = ";
        if (auto range = fs.GetRange()) {
            if (auto start = range->GetStart()) {
                start->PrintSSAName(os_);
            } else {
                os_ << "<null>";
            }
            os_ << " to ";
            if (auto end = range->GetEnd()) {
                end->PrintSSAName(os_);
            } else {
                os_ << "<null>";
            }
            os_ << " step ";
            if (auto step = range->GetStep()) {
                step->PrintSSAName(os_);
            } else {
                os_ << "<null>";
            }
        } else {
            os_ << "<null> to <null> step <null>";
        }

        // Print iter_args if present.
        const auto& iterArgs = fs.IterArgs();
        if (!iterArgs.empty()) {
            os_ << "\n";
            PrintIndentWithExtra(2);
            os_ << "iter_args(";
            for (size_t i = 0; i < iterArgs.size(); ++i) {
                const auto& arg = iterArgs[i];
                // Use value's SSA name as iter_arg identifier
                if (arg.value) {
                    arg.value->PrintSSAName(os_);
                    os_ << " = ";
                } else {
                    os_ << "<no-value> = ";
                }
                if (arg.initValue) {
                    arg.initValue->PrintValue(os_);
                } else {
                    os_ << "<null> : <unknown>";
                }
                if (i + 1 < iterArgs.size()) {
                    os_ << ", ";
                }
            }
            os_ << ")";
        }
        // Print attributes if present.
        const auto& attrs = fs.Attributes();
        if (!attrs.empty()) {
            os_ << "\n";
            PrintIndentWithExtra(2);
            os_ << "attributes {";
            bool first = true;
            for (const auto& kv : attrs) {
                if (!first) {
                    os_ << ", ";
                }
                os_ << kv.first << " = " << kv.second;
                first = false;
            }
            os_ << "}";
        }

        os_ << " {\n";

        // Use Visitor to print loop body
        Indent();
        if (auto compound = fs.GetCompound()) {
            VisitCompoundStatement(*compound);
        }
        Dedent();

        PrintIndent();
        os_ << "}\n";
    }

    void VisitYieldStatement(YieldStatement& ys) {
        PrintIndent();
        os_ << "statement.yield";
        const auto& values = ys.Values();
        if (!values.empty()) {
            os_ << " ";
            PrintValueList(values);
        }
        os_ << "\n";
    }

    void VisitReturnStatement(ReturnStatement& rs) {
        PrintIndent();
        os_ << "statement.return";
        const auto& values = rs.Values();
        if (!values.empty()) {
            os_ << " ";
            PrintValueList(values);
        }
        os_ << "\n";
    }

private:
    // ===== helpers =====
    void PrintIndent() {
        for (int i = 0; i < indent_; ++i) os_ << "  ";
    }

    void PrintIndentWithExtra(int extra) {
        for (int i = 0; i < indent_ + extra; ++i) os_ << "  ";
    }

    void Indent() { indent_++; }
    void Dedent() { indent_--; }

    void PrintValueList(const std::vector<ValuePtr>& vals) {
        for (size_t i = 0; i < vals.size(); ++i) {
            if (i > 0) os_ << ", ";
            if (vals[i]) {
                vals[i]->PrintSSAName(os_);
            } else {
                os_ << "<null>";
            }
        }
    }

    void PrintAttributes(const AttributeMap& attrs, const std::string& prefix) {
        for (const auto& kv : attrs) {
            os_ << prefix << kv.first << " = " << kv.second << "\n";
        }
    }

    const char* ToString(FunctionKind kind) {
        switch (kind) {
        case FunctionKind::ControlFlow:
            return "control_flow";
        case FunctionKind::DataFlow:
            return "data_flow";
        case FunctionKind::Block:
            return "block";
        default:
            return "unknown";
        }
    }

private:
    std::ostream& os_;
    IROpcodeFormatter formatter_;
    int indent_{0};
};

} // namespace pto
