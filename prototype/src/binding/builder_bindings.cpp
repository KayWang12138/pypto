// PTO-IR prototype: Builder bindings for Python.
// All comments must remain in English for consistency.

#include "bindings.h"
#include "ir/program.h"
#include "ir/scope.h"
#include "ir/builder/ir_builder.h"
#include "ir/builder/guard.h"
#include "ir/function.h"
#include "ir/statement.h"

#include <pybind11/stl.h>
#include <sstream>

namespace pto {

// Helper function to bind ProgramModule
static void BindProgramModule(py::module_ &m) {
    py::class_<ProgramModule, std::shared_ptr<ProgramModule>>(m, "ProgramModule")
        .def(py::init<std::string>(), py::arg("name"))
        .def("SetProgramEntry", &ProgramModule::SetProgramEntry, py::arg("programEntry"))
        .def("GetProgramEntry", &ProgramModule::GetProgramEntry)
        .def("AddFunction", &ProgramModule::AddFunction, py::arg("function"))
        .def("GetFunctions", &ProgramModule::GetFunctions)
        .def("Print", [](const ProgramModule &m) {
            std::ostringstream os;
            m.Print(os, 0);
            return os.str();
        })
        .def("__repr__", [](const ProgramModule &m) {
            std::ostringstream os;
            m.Print(os, 0);
            return os.str();
        });
}

// Helper function to bind Scope
static void BindScope(py::module_ &m) {
    py::class_<Scope, std::shared_ptr<Scope>>(m, "Scope")
        .def("SetParent", &Scope::SetParent, py::arg("parent"))
        .def("GetParent", &Scope::GetParent,
             py::return_value_policy::reference_internal)
        .def("AddStatement", &Scope::AddStatement, py::arg("stmt"))
        .def("GetStatements",
             py::overload_cast<>(&Scope::GetStatements, py::const_))
        .def("FindValue", &Scope::FindValue, py::arg("name"))
        .def("SetEnvVar", &Scope::SetEnvVar, py::arg("name"), py::arg("value"))
        .def("GetEnvVar", &Scope::GetEnvVar, py::arg("name"));
}

// Helper function to bind ScopeGuard
static void BindScopeGuard(py::module_ &m) {
    py::class_<ScopeGuard, std::shared_ptr<ScopeGuard>>(m, "ScopeGuard")
        // Note: ScopeGuard should not be directly constructed from Python
        // It's only returned by IRBuilder methods
        .def("__enter__", [](std::shared_ptr<ScopeGuard> self) {
            return self;
        })
        .def("__exit__", [](std::shared_ptr<ScopeGuard> self, py::object /*exc_type*/,
                           py::object /*exc_value*/, py::object /*traceback*/) {
            // ScopeGuard will be destroyed when Python context exits
            // The C++ destructor will restore the previous scope
            return false;  // Don't suppress exceptions
        });
}

// Helper function to bind IRBuilder
static void BindIRBuilder(py::module_ &m) {
    py::class_<IRBuilder>(m, "IRBuilder")
        .def(py::init<>())
        .def(py::init<ProgramModule*>(), py::arg("module"))
        .def("SetModule", &IRBuilder::SetModule, py::arg("module"))
        
        // Function management
        .def("CreateFunction",
             &IRBuilder::CreateFunction,
             py::arg("name"),
             py::arg("kind"),
             py::arg("signature"),
             py::arg("setAsEntry") = false,
             py::return_value_policy::reference_internal)
        .def("SetCurrentFunction", &IRBuilder::SetCurrentFunction, py::arg("function"))
        .def("GetCurrentFunction", &IRBuilder::GetCurrentFunction,
             py::return_value_policy::reference_internal)
        .def("GetCurrentScope", &IRBuilder::GetCurrentScope,
             py::return_value_policy::reference_internal)
        .def("GetCurrentBlock", &IRBuilder::GetCurrentBlock,
             py::return_value_policy::reference_internal)
        
        // Insertion point
        .def("GetOrCreateActiveBlock", &IRBuilder::GetOrCreateActiveBlock,
             py::return_value_policy::reference_internal)
        
        // Value creation
        .def("AddToScope", &IRBuilder::AddToScope, py::arg("value"))
        .def("CreateTensor",
             [](IRBuilder &builder, const std::vector<Scalar>& shape,
                DataType dtype, const std::string& name) {
                 return builder.CreateTensor(shape, dtype, name);
             },
             py::arg("shape"),
             py::arg("dtype"),
             py::arg("name") = "")
        .def("CreateTile",
             [](IRBuilder &builder, const std::vector<size_t>& shape,
                DataType dtype, const std::string& name) {
                 return builder.CreateTile(shape, dtype, name);
             },
             py::arg("shape"),
             py::arg("dtype"),
             py::arg("name") = "")
        .def("CreateScalar",
             &IRBuilder::CreateScalar,
             py::arg("dtype"),
             py::arg("name") = "")
        .def("CreateConst",
             static_cast<std::shared_ptr<Scalar>(IRBuilder::*)(int64_t, std::string)>
                 (&IRBuilder::CreateConst),
             py::arg("value"),
             py::arg("name") = "")
        .def("CreateConst",
             static_cast<std::shared_ptr<Scalar>(IRBuilder::*)(double, std::string)>
                 (&IRBuilder::CreateConst),
             py::arg("value"),
             py::arg("name") = "")

        // DuplicateValue
        .def("DuplicateValue", &IRBuilder::DuplicateValue,
             py::arg("old"))
        
        // Op emission
        .def("Emit", &IRBuilder::Emit, py::arg("op"),
             py::return_value_policy::reference_internal)
        .def("CreateOp",
             [](IRBuilder &builder,
                Opcode opcode,
                const ValuePtrs& inputs,
                std::shared_ptr<OpPayload> payload,
                const std::string& name) {
                 return builder.CreateOp(opcode, inputs, payload, name);
             },
             py::arg("opcode"),
             py::arg("inputs"),
             py::arg("payload") = nullptr,
             py::arg("name") = "")
        
        // Statement building
        .def("CreateBlockStmt", &IRBuilder::CreateBlockStmt,
             py::return_value_policy::reference_internal)
        .def("CreateForStmt",
             &IRBuilder::CreateForStmt,
             py::arg("iterationVar"),
             py::arg("start"),
             py::arg("end"),
             py::arg("step"),
             py::return_value_policy::reference_internal)
        .def("CreateIfStmt",
             &IRBuilder::CreateIfStmt,
             py::arg("condition"),
             py::return_value_policy::reference_internal)
        .def("CreateYield", &IRBuilder::CreateYield,
             py::arg("values"),
             py::return_value_policy::reference_internal)
        .def("CreateReturn", &IRBuilder::CreateReturn,
             py::arg("values"),
             py::return_value_policy::reference_internal)
        .def("CreateCall", &IRBuilder::CreateCall, 
             py::arg("callee"),
             py::arg("args"),
             py::arg("res"),
             py::arg("pos"),
             py::return_value_policy::reference_internal)
     
        // Interface for Modifying IR 
        .def("RemoveBlockStatement", &IRBuilder::RemoveBlockStatement, 
             py::arg("st"))
        .def("ReplaceValueInBlock", &IRBuilder::ReplaceValueInBlock,
             py::arg("st"),
             py::arg("oldValue"),
             py::arg("newValue"))
        .def("ReplaceOpOperand", &IRBuilder::ReplaceOpOperand,
             py::arg("op"),
             py::arg("oldOperand"),
             py::arg("newOperand"))

        // Enter nested scopes
        .def("EnterFunctionBody", &IRBuilder::EnterFunctionBody,
             py::arg("function"))
        .def("EnterForBody", &IRBuilder::EnterForBody,
             py::arg("forStmt"))
        .def("EnterIfThen", &IRBuilder::EnterIfThen,
             py::arg("ifStmt"))
        .def("EnterIfElse", &IRBuilder::EnterIfElse,
             py::arg("ifStmt"))
        
        // Environment table management
        .def("SetEnvVar", &IRBuilder::SetEnvVar,
             py::arg("name"),
             py::arg("value"))
        .def("GetEnvVar", &IRBuilder::GetEnvVar,
             py::arg("name"))
        .def("GetCurrentEnv", &IRBuilder::GetCurrentEnv)
        .def("ExitIfStatement", &IRBuilder::ExitIfStatement,
             py::arg("ifStmt"))
        .def("ExitForStatement", &IRBuilder::ExitForStatement,
             py::arg("forStmt"));
}

// Main builder bindings function
void BindBuilderBindings(py::module_ &m) {
    // ProgramModule, Scope, ScopeGuard, and IRBuilder bindings
    BindProgramModule(m);
    BindScope(m);
    BindScopeGuard(m);
    BindIRBuilder(m);
}

}  // namespace pto

