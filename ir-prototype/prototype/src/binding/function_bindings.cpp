// PTO-IR prototype: Function bindings for Python.
// All comments must remain in English for consistency.

#include "bindings.h"
#include "ir/function.h"
#include "ir/scope.h"

#include <pybind11/stl.h>
#include <sstream>

namespace pto {

// Helper function to expose FunctionKind enum.
static void BindFunctionKind(py::module_ &m) {
    py::enum_<FunctionKind>(m, "FunctionKind")
        .value("ControlFlow", FunctionKind::ControlFlow)
        .value("DataFlow", FunctionKind::DataFlow)
        .value("Kernel", FunctionKind::Kernel)
        .export_values();
}

// Helper function to expose FunctionSignature struct.
static void BindFunctionSignature(py::module_ &m) {
    py::class_<FunctionSignature>(m, "FunctionSignature")
        .def(py::init<>())
        .def_readwrite("arguments", &FunctionSignature::arguments)
        .def_readwrite("results", &FunctionSignature::results);
}

// Helper function to expose Function class.
static void BindFunction(py::module_ &m) {
    py::class_<Function, std::shared_ptr<Function>>(m, "Function")
        .def(py::init<const std::string &, FunctionKind, const FunctionSignature &>(),
             py::arg("name"),
             py::arg("kind"),
             py::arg("signature"))
        .def("GetKind", &Function::GetKind)
        .def("GetSignature", &Function::GetSignature,
             py::return_value_policy::reference_internal)
        // Note: Body() and GetScope() are not exposed because they return references
        // to containers with shared_ptr elements, which cannot be copied by pybind11.
        // Use AddStatement() to add statements instead.
        .def("GetScope", (Scope& (Function::*)()) &Function::GetScope,
             py::return_value_policy::reference_internal)
        .def("GetInputScope", (Scope& (Function::*)()) &Function::GetInputScope,
             py::return_value_policy::reference_internal)
        .def("AddStatement", &Function::AddStatement)
        .def("Print", [](const Function &f) {
            std::ostringstream os;
            f.Print(os, 0);
            return os.str();
        })
        .def("__repr__", [](const Function &f) {
            std::ostringstream os;
            f.Print(os, 0);
            return os.str();
        });
}

// Main function bindings function
void BindFunctionBindings(py::module_ &m) {
    // FunctionKind enum, FunctionSignature struct, and Function class bindings.
    BindFunctionKind(m);
    BindFunctionSignature(m);
    BindFunction(m);
}

}  // namespace pto

