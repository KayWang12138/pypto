// PTO-IR prototype: operation-level IR structures.
// All comments must remain in English for consistency.

#pragma once

#include "ir/utils_defop.h"
#include "ir/utils.h"
#include "ir/value.h"
#include "ir/opcode.h"
#include "ir/operation_base.h"

#include <memory>
#include <ostream>
#include <string>

namespace pto {

#define DEFOP DEFOP_CLASS
#include "operation.def"
#undef DEFOP

} // namespace pto


