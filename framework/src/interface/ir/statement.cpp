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

} // namespace pto


