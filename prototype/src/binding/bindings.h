// PTO-IR prototype: Python bindings declarations.
// All comments must remain in English for consistency.

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pto {

// Type bindings: DataType, TileOpFormat, Scalar, Value, enums
void BindTypeBindings(py::module_ &m);

// Tensor bindings: Tensor class and helper functions
void BindTensorBindings(py::module_ &m);

// Function bindings: FunctionKind, FunctionSignature, Function
void BindFunctionBindings(py::module_ &m);

// Statement bindings: Statement hierarchy and helper functions
void BindStatementBindings(py::module_ &m);

// Operation bindings: Opcode, operations, and operation helpers
void BindOperationBindings(py::module_ &m);

// Pass bindings: Result, Pass, PassManager
void BindPassBindings(py::module_ &m);

// Builder bindings: ProgramModule, Scope, ScopeGuard, IRBuilder
void BindBuilderBindings(py::module_ &m);

}  // namespace pto

