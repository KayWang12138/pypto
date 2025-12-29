// PTO-IR prototype: Statement bindings for Python.
// All comments must remain in English for consistency.

#include "bindings.h"
#include "ir/statement.h"
#include "ir/function.h"

#include <pybind11/stl.h>
#include <sstream>

namespace pto {

// Helper function to bind Statement base class.
static void BindStatement(py::module_ &m) {
    py::class_<Statement, std::shared_ptr<Statement>>(m, "Statement")
        .def("GetKind", &Statement::GetKind)
        .def("Print", [](const Statement &s) {
            std::ostringstream os;
            s.Print(os, 0);
            return os.str();
        });
}

// Helper function to bind BlockStatement.
static void BindBlockStatement(py::module_ &m) {
    py::class_<BlockStatement, Statement, std::shared_ptr<BlockStatement>>(m, "BlockStatement")
        .def(py::init<>())
        .def("Operations", [](BlockStatement &stmt) -> py::list {
            py::list ops;
            for (auto &op : stmt.Operations()) {
                ops.append(op.get());
            }
            return ops;
        }, py::return_value_policy::reference_internal)

        .def("__repr__", [](const BlockStatement &st) {
            std::ostringstream os;
            st.Print(os, 0);
            return os.str();
        });
}

// Helper function to bind LoopRange.
static void BindLoopRange(py::module_ &m) {
    py::class_<LoopRange, std::shared_ptr<LoopRange>>(m, "LoopRange")
        .def(py::init<std::shared_ptr<Scalar>, std::shared_ptr<Scalar>, std::shared_ptr<Scalar>>(),
             py::arg("start"),
             py::arg("end"),
             py::arg("step"))
        .def("GetStart", &LoopRange::GetStart,
             py::return_value_policy::reference_internal)
        .def("GetEnd", &LoopRange::GetEnd,
             py::return_value_policy::reference_internal)
        .def("GetStep", &LoopRange::GetStep,
             py::return_value_policy::reference_internal);
}

// Helper function to bind ForStatement.
static void BindForStatement(py::module_ &m) {
    py::class_<ForStatement, Statement, std::shared_ptr<ForStatement>>(m, "ForStatement")
        .def(py::init<std::shared_ptr<Scalar>, std::shared_ptr<Scalar>, std::shared_ptr<Scalar>, std::shared_ptr<Scalar>>(),
             py::arg("iterationVar"),
             py::arg("start"),
             py::arg("end"),
             py::arg("step"))
        .def("GetIterationVar", &ForStatement::GetIterationVar,
             py::return_value_policy::reference_internal)
        .def("GetStart", &ForStatement::GetStart,
             py::return_value_policy::reference_internal)
        .def("GetEnd", &ForStatement::GetEnd,
             py::return_value_policy::reference_internal)
        .def("GetStep", &ForStatement::GetStep,
             py::return_value_policy::reference_internal)
        .def("AddIterArg", &ForStatement::AddIterArg,
             py::arg("initValue"),
             "Add a loop-carried accumulator argument with the given initial value.")
        .def("IterArgs", [](ForStatement &stmt) -> py::list {
            py::list args;
            for (auto &arg : stmt.IterArgs()) {
                py::dict arg_dict;
                arg_dict["initValue"] = arg.initValue;
                arg_dict["value"] = arg.value;
                args.append(arg_dict);
            }
            return args;
        }, py::return_value_policy::reference_internal)
        .def("Body", 
            py::overload_cast<>(&ForStatement::Body, py::const_))
        .def("Print", [](ForStatement &stmt) {
            std::ostringstream os;
            stmt.Print(os, 0);
            return os.str();
        })
        
        .def("__repr__", [](const ForStatement &st) {
            std::ostringstream os;
            st.Print(os, 0);
            return os.str();
        });
}

// Helper function to bind IfStatement.
static void BindIfStatement(py::module_ &m) {
    py::class_<IfStatement, Statement, std::shared_ptr<IfStatement>>(m, "IfStatement")
        .def(py::init<const std::string &>(),
             py::arg("condition"))
        .def("GetCondition", &IfStatement::GetCondition,
             py::return_value_policy::reference_internal)
             
        .def("__repr__", [](const IfStatement &st) {
            std::ostringstream os;
            st.Print(os, 0);
            return os.str();
        });
}

// Helper function to bind CallStatement
static void BindCallStatement(py::module_ &m) {
    py::class_<CallStatement, Statement, std::shared_ptr<CallStatement>>(m, "CallStatement")
        .def(py::init<std::string, ValuePtrs&, ValuePtrs&>(),
             py::arg("callee"), py::arg("args"), py::arg("res"))
        .def("GetKind", &CallStatement::GetKind)
        .def("GetCallee", &CallStatement::GetCallee, py::return_value_policy::reference_internal)
        .def("Arguments", (ValuePtrs& (CallStatement::*)()) &CallStatement::Arguments, py::return_value_policy::reference_internal)
        .def("Arguments", (const ValuePtrs& (CallStatement::*)() const) &CallStatement::Arguments, py::return_value_policy::reference_internal)
        .def("Results", (ValuePtrs& (CallStatement::*)()) &CallStatement::Results, py::return_value_policy::reference_internal)
        .def("Results", (const ValuePtrs& (CallStatement::*)() const) &CallStatement::Results, py::return_value_policy::reference_internal)
        .def("Print", [](const CallStatement& stmt, int indent = 0) -> std::string {
            std::ostringstream os;
            stmt.Print(os, indent);
            return os.str();
        }, py::arg("indent") = 0)
        
        .def("__repr__", [](const CallStatement &st) {
            std::ostringstream os;
            st.Print(os, 0);
            return os.str();
        });
}

// Helper function to bind ReturnStatement
static void BindReturnStatement(py::module_ &m) {
    py::class_<ReturnStatement, Statement, std::shared_ptr<ReturnStatement>>(m, "ReturnStatement")
        .def("GetKind", &ReturnStatement::GetKind)
        .def("Values", (ValuePtrs& (ReturnStatement::*)()) &ReturnStatement::Values, 
             py::return_value_policy::reference_internal)
        .def("Values", (const ValuePtrs& (ReturnStatement::*)() const) &ReturnStatement::Values, 
             py::return_value_policy::reference_internal)
        .def("Print", [](const ReturnStatement& stmt, int indent = 0) -> std::string {
            std::ostringstream os;
            stmt.Print(os, indent);
            return os.str();
        }, py::arg("indent") = 0)
        
        .def("__repr__", [](const ReturnStatement &st) {
            std::ostringstream os;
            st.Print(os, 0);
            return os.str();
        });
}

// Helper function to bind YieldStatement
static void BindYieldStatement(py::module_ &m) {
    py::class_<YieldStatement, Statement, std::shared_ptr<YieldStatement>>(m, "YieldStatement")
        .def("GetKind", &YieldStatement::GetKind)
        .def("Values", (ValuePtrs& (YieldStatement::*)()) &YieldStatement::Values, 
             py::return_value_policy::reference_internal)
        .def("Values", (const ValuePtrs& (YieldStatement::*)() const) &YieldStatement::Values, 
             py::return_value_policy::reference_internal)
        .def("Print", [](const YieldStatement& stmt, int indent = 0) -> std::string {
            std::ostringstream os;
            stmt.Print(os, indent);
            return os.str();
        }, py::arg("indent") = 0)
        
        .def("__repr__", [](const YieldStatement &st) {
            std::ostringstream os;
            st.Print(os, 0);
            return os.str();
        });
}

// Helper function to create ForStatement and add it to a function.
static ForStatement* CreateForStatement(
    std::shared_ptr<Scalar> iterationVar,
    std::shared_ptr<Scalar> start,
    std::shared_ptr<Scalar> end,
    std::shared_ptr<Scalar> step,
    Function* func) {
    if (func == nullptr) {
        throw std::runtime_error("CreateForStatement: func is null");
    }
    
    auto forStmt = std::make_shared<ForStatement>(
        std::move(iterationVar), std::move(start), std::move(end), std::move(step));
    auto* ref = forStmt.get();
    func->AddStatement(std::move(forStmt));
    return ref;
}

// Helper function to create IfStatement and add it to a function.
static IfStatement* CreateIfStatement(const std::string &condition, Function* func) {
    if (func == nullptr) {
        throw std::runtime_error("CreateIfStatement: func is null");
    }

    auto ifStmt = std::make_shared<IfStatement>(condition);
    auto* ref = ifStmt.get();
    func->AddStatement(std::move(ifStmt));
    return ref;
}

// Helper function to add a BlockStatement to ForStatement's Body.
static BlockStatement* AddBlockToForStatement(ForStatement* forStmt) {
    if (forStmt == nullptr) {
        throw std::runtime_error("AddBlockToForStatement: forStmt is null");
    }
    
    auto block = std::make_shared<BlockStatement>();
    auto* ref = block.get();
    forStmt->Body().push_back(std::move(block));
    return ref;
}

// Helper function to add a BlockStatement to Function's Body.
static BlockStatement* AddBlockToFunc(Function* func) {
    if (func == nullptr) {
        throw std::runtime_error("AddBlockToFunc: func is null");
    }
    
    auto block = std::make_shared<BlockStatement>();
    auto* ref = block.get();
    func->Body().push_back(std::move(block));
    return ref;
}

// Helper function to add a BlockStatement to IfStatement's then branch.
static BlockStatement* AddThenBlockToIf(IfStatement* ifStmt) {
    if (ifStmt == nullptr) {
        throw std::runtime_error("AddThenBlockToIf: ifStmt is null");
    }

    auto block = std::make_shared<BlockStatement>();
    auto* ref = block.get();
    ifStmt->ThenBranch().push_back(std::move(block));
    return ref;
}

// Helper function to add a BlockStatement to IfStatement's else branch.
static BlockStatement* AddElseBlockToIf(IfStatement* ifStmt) {
    if (ifStmt == nullptr) {
        throw std::runtime_error("AddElseBlockToIf: ifStmt is null");
    }

    auto block = std::make_shared<BlockStatement>();
    auto* ref = block.get();
    ifStmt->ElseBranch().push_back(std::move(block));
    return ref;
}

// Main statement bindings function
void BindStatementBindings(py::module_ &m) {
    // Statement base class binding.
    BindStatement(m);

    // LoopRange binding.
    BindLoopRange(m);

    // BlockStatement binding.
    BindBlockStatement(m);

    // ForStatement binding.
    BindForStatement(m);

    // IfStatement binding.
    BindIfStatement(m);

    // CallStatement binding.
    BindCallStatement(m);

    // ReturnStatement binding.
    BindReturnStatement(m);

    // YieldStatement binding.
    BindYieldStatement(m);

    // ForStatement creation function.
    m.def("CreateForStatement", &CreateForStatement,
          py::arg("iterationVar"),
          py::arg("start"),
          py::arg("end"),
          py::arg("step"),
          py::arg("func"),
          py::return_value_policy::reference);
    
    // Helper function to add BlockStatement to ForStatement's Body.
    m.def("AddBlockToForStatement", &AddBlockToForStatement,
          py::arg("forStmt"),
          py::return_value_policy::reference);
    
    // Helper function to add BlockStatement to Function's Body.
    m.def("AddBlockToFunc", &AddBlockToFunc,
          py::arg("func"),
          py::return_value_policy::reference);

    // IfStatement creation function.
    m.def("CreateIfStatement", &CreateIfStatement,
          py::arg("condition"),
          py::arg("func"),
          py::return_value_policy::reference);

    // Helper functions to add BlockStatement to IfStatement branches.
    m.def("AddThenBlockToIf", &AddThenBlockToIf,
          py::arg("ifStmt"),
          py::return_value_policy::reference);
    m.def("AddElseBlockToIf", &AddElseBlockToIf,
          py::arg("ifStmt"),
          py::return_value_policy::reference);
}

}  // namespace pto

