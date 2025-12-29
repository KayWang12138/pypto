// PTO-IR prototype: Pass bindings for Python.
// All comments must remain in English for consistency.

#include "bindings.h"
#include "pass/pass.h"
#include "pass/liveness.h"
#include "pass/pass_manager.h"
#include "ir/function.h"
#include "ir/statement.h"

#include <pybind11/attr.h>
#include <pybind11/stl.h>

namespace pto {

// Trampoline class for python inherit Pass
class PyPass : public Pass {
public:
    using Pass::Pass;

    Result PreCheck(ProgramModule& m) override {
        PYBIND11_OVERRIDE(Result, Pass, PreCheck, m);
    }

    Result RunOnModule(ProgramModule& m) override {
        PYBIND11_OVERRIDE(Result, Pass, RunOnModule, m);
    }

    Result RunOnFunction(std::shared_ptr<Function> func) override {
        PYBIND11_OVERRIDE_PURE(Result, Pass, RunOnFunction, func);
    }
    
    Result RunOnStatement(StatementPtr stmt) override {
        PYBIND11_OVERRIDE_PURE(Result, Pass, RunOnStatement, stmt);
    }
    
    Result RunOnOperation(OperationPtr op) override {
        PYBIND11_OVERRIDE_PURE(Result, Pass, RunOnOperation, op);
    }

    Result PostCheck(ProgramModule& m) override {
        PYBIND11_OVERRIDE(Result, Pass, PostCheck, m);
    }
};

static void BindResult(py::module_ &m) {
    py::enum_<Result>(m, "Result")
        .value("Success", Result::Success)
        .value("Failure", Result::Failure)
        .export_values();
}

static void BindPass(py::module_ &m) {
    py::class_<Pass, PyPass, std::shared_ptr<Pass>>(m, "Pass")
        .def(py::init<const std::string&>(), py::arg("name"))
        .def("Run", &Pass::Run, py::arg("module"),
             py::call_guard<py::gil_scoped_release>()) 
        
        .def("PreCheck", &Pass::PreCheck, py::arg("module"))
        .def("RunOnModule", &Pass::RunOnModule, py::arg("module"))
        .def("RunOnFunction", &Pass::RunOnFunction, py::arg("func"))
        .def("RunOnStatement", &Pass::RunOnStatement, py::arg("stmt"))
        .def("RunOnOperation", &Pass::RunOnOperation, py::arg("op"))
        .def("PostCheck", &Pass::PostCheck, py::arg("module"))
        .def("GetName", &Pass::GetName)
        
        // Python method
        .def("__str__", [](Pass &pass) {
            return "Pass('" + pass.GetName() + "')";
        })
        .def("__repr__", [](Pass &pass) {
            return "<Pass '" + pass.GetName() + "'>";
        });
}

static void BindPassManager(py::module_ &m) {
    py::class_<PassManager, std::shared_ptr<PassManager>>(m, "PassManager")
        .def_static("Instance", &PassManager::Instance, 
                   py::return_value_policy::reference)
        
        // Use keep alive to keep python pass object alive,
        // ensuring python pass override method can be call correctly.
        .def("AddPass", &PassManager::AddPass, py::arg("pass"),
             py::keep_alive<1, 2>())
        
        // Overloaded Run methods
        .def("Run", 
             static_cast<Result (PassManager::*)(ProgramModule&, bool)>(&PassManager::Run),
             py::arg("module"), py::arg("print") = false,
             py::call_guard<py::gil_scoped_release>())
        
        .def("Run", 
             static_cast<Result (PassManager::*)(std::shared_ptr<Function>, bool)>(&PassManager::Run),
             py::arg("func"), py::arg("print") = false,
             py::call_guard<py::gil_scoped_release>())
    
        .def("clearPasses", &PassManager::Clear)

        .def("__del__", [](const PassManager&) {});
}

static void BindLivenessAnalyzer(py::module_ &m) {
    py::class_<LivenessAnalyzer>(m, "LivenessAnalyzer")
        .def(py::init<>())
        .def("RunLivenessAnalysis", &LivenessAnalyzer::RunLivenessAnalysis, py::arg("func"))
        .def("GetLiveIn", &LivenessAnalyzer::GetLiveIn, 
             py::arg("stmt"), 
             py::return_value_policy::reference_internal)
        .def("GetLiveOut", &LivenessAnalyzer::GetLiveOut, 
             py::arg("stmt"), 
             py::return_value_policy::reference_internal)
        .def("GetUse", &LivenessAnalyzer::GetUse,
             py::arg("stmt"), 
             py::return_value_policy::reference_internal)
        .def("GetDef", &LivenessAnalyzer::GetDef,
             py::arg("stmt"), 
             py::return_value_policy::reference_internal);
}

// Main pass bindings function
void BindPassBindings(py::module_ &m) {
    BindResult(m);
    BindPass(m);
    BindPassManager(m);
    BindLivenessAnalyzer(m);
}

}  // namespace pto

