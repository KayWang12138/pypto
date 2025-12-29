// PTO-IR prototype: function-level IR structures implementation.

#include "ir/function.h"

namespace pto {

Function::Function(std::string name, FunctionKind kind, FunctionSignature signature)
    : Object(ObjectType::Function, std::move(name)),
      kind_(kind),
      signature_(std::move(signature)) {
    // Register function arguments into the dedicated input scope so that
    // Function::scope_ can see them via GetAncestorValues().
    for (const auto& arg : signature_.arguments) {
        if (arg) {
            // Use SSA name as the key in environment table
            input_scope_.SetEnvVar(arg->GetSSAName(), arg);
        }
    }

    // Make the function body scope a child of the input scope.
    scope_.SetParent(&input_scope_);
}

void Function::AddStatement(StatementPtr stmt) {
    scope_.AddStatement(std::move(stmt));
}

static const char* toString(FunctionKind kind) {
    switch (kind) {
    case FunctionKind::ControlFlow:
        return "control_flow";
    case FunctionKind::DataFlow:
        return "data_flow";
    case FunctionKind::Kernel:
        return "kernel";
    }
    return "unknown";
}

static void PrintAttributes(std::ostream& os, const AttributeMap& attrs, const std::string& prefix) {
    for (const auto& kv : attrs) {
        os << prefix << kv.first << " = " << kv.second << "\n";
    }
}

void Function::Print(std::ostream& os, int indent) const {
    // Print indentation for the function header.
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
    os << "func.func " << GetPrefixedName() << "(";
    for (size_t i = 0; i < signature_.arguments.size(); ++i) {
        const auto& arg = signature_.arguments[i];
        if (arg) {
            os << arg->GetSSAName() << ": ";
            arg->Print(os, 0);
        }
        if (i + 1 < signature_.arguments.size()) {
            os << ", ";
        }
    }
    os << ")";

    if (!signature_.results.empty()) {
        os << " -> (";
        for (size_t i = 0; i < signature_.results.size(); ++i) {
            if (signature_.results[i]) {
                signature_.results[i]->Print(os, 0);
            }
            if (i + 1 < signature_.results.size()) {
                os << ", ";
            }
        }
        os << ")";
    }

    os << " {\n";

    // Print structured statement body if present.
    for (const auto& stmt : scope_.GetStatements()) {
        if (stmt) {
            stmt->Print(os, indent + 1);
        }
    }

    // Closing brace.
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
    os << "}\n";

    // Function attributes can be printed as separate lines following MLIR style.
    // Example:
    // func.attr private = true
    // func.attr inline = true
    std::string attrPrefix;
    for (int i = 0; i < indent; ++i) {
        attrPrefix += "  ";
    }
    attrPrefix += "func.attr ";
    PrintAttributes(os, attributes_, attrPrefix);

    // Optionally print a comment giving the function kind.
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
    os << "// func.kind = " << toString(kind_) << "\n";
}

std::ostream& operator<<(std::ostream& os, const Function& func) {
    func.Print(os, 0);
    return os;
}

} // namespace pto


