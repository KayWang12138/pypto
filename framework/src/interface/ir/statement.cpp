// PTO-IR prototype: statement-level IR structures implementation.

#include "ir/statement.h"
#include "ir/utils.h"

#include <ostream>

namespace pto {

void BlockStatement::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    os << "statement.block {\n";

    // Print linear operations.
    for (const auto& op : operations_) {
        if (op) {
            op->Print(os, indent + 2);
        }
    }

    PrintIndent(os, indent);
    os << "}\n";
}

void ReturnStatement::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    os << "statement.return";
    if (!values_.empty()) {
        os << " ";
        for (size_t i = 0; i < values_.size(); ++i) {
            if (values_[i]) {
                os << values_[i]->GetSSAName();
            }
            if (i + 1 < values_.size()) {
                os << ", ";
            }
        }
    }
    os << "\n";
}

} // namespace pto


