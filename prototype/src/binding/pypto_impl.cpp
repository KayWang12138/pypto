// PTO-IR prototype: Python bindings for core IR types.
// All comments must remain in English for consistency.

#include "bindings.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pypto_impl, m) {
    m.doc() = "Python bindings for PTO-IR core types";

    // DataType, TileOpFormat, Scalar, Value, and enums (CastMode, ReLuType, ReduceKind)
    pto::BindTypeBindings(m);

    // Tensor class and helper functions (SetTensorData, GetTensorData, GetInputShape)
    pto::BindTensorBindings(m);

    // FunctionKind, FunctionSignature, Function
    pto::BindFunctionBindings(m);

    // Statement hierarchy and helper functions
    pto::BindStatementBindings(m);

    // Opcode, operations, and operation helpers
    pto::BindOperationBindings(m);

    // Result, Pass, PassManager
    pto::BindPassBindings(m);

    // ProgramModule, Scope, ScopeGuard, IRBuilder
    pto::BindBuilderBindings(m);
}

