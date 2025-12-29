// PTO-IR prototype: Type bindings for Python.
// All comments must remain in English for consistency.

#include "bindings.h"
#include "ir/type.h"
#include "ir/op/op_payload.h"

#include <pybind11/stl.h>
#include <sstream>

namespace pto {

// Helper function to expose DataType as Python DataType enum.
static void BindDataType(py::module_ &m) {
    py::enum_<DataType>(m, "DataType")
        .value("DT_BOOL", DataType::BOOL)

        .value("DT_INT4", DataType::INT4)
        .value("DT_INT8", DataType::INT8)
        .value("DT_INT16", DataType::INT16)
        .value("DT_INT32", DataType::INT32)
        .value("DT_INT64", DataType::INT64)

        .value("DT_UINT8", DataType::UINT8)
        .value("DT_UINT16", DataType::UINT16)
        .value("DT_UINT32", DataType::UINT32)
        .value("DT_UINT64", DataType::UINT64)

        .value("DT_FP8", DataType::FP8)
        .value("DT_FP16", DataType::FP16)
        .value("DT_BF16", DataType::BF16)
        .value("DT_FP32", DataType::FP32)
        .value("DT_DOUBLE", DataType::FP64)

        .value("DT_HF4", DataType::HF4)
        .value("DT_HF8", DataType::HF8)

        .value("DT_BOTTOM", DataType::BOTTOM)
        .export_values();
}

// Helper function to expose TileOpFormat as Python TileOpFormat enum.
static void BindTileOpFormat(py::module_ &m) {
    py::enum_<TileOpFormat>(m, "TileOpFormat")
        .value("TILEOP_ND", TileOpFormat::TILEOP_ND)
        .value("TILEOP_NZ", TileOpFormat::TILEOP_NZ)
        .export_values();
}

// Helper function to expose Scalar and ScalarValueKind.
static void BindScalar(py::module_ &m) {
    py::enum_<ScalarValueKind>(m, "ScalarValueKind")
        .value("Constant", ScalarValueKind::Constant)
        .value("Symbolic", ScalarValueKind::Symbolic)
        .export_values();

    py::class_<Scalar, Value, std::shared_ptr<Scalar>>(m, "Scalar")
        .def(py::init<>())
        .def(py::init<DataType, const std::string &, ScalarValueKind>(),
             py::arg("type"),
             py::arg("name") = "",
             py::arg("valueKind") = ScalarValueKind::Symbolic)
        .def(py::init<const std::string &, const std::string &, ScalarValueKind>(),
             py::arg("typeName"),
             py::arg("name") = "",
             py::arg("valueKind") = ScalarValueKind::Symbolic)
        .def(py::init<bool, const std::string &>(),
             py::arg("value"),
             py::arg("name") = "")
        .def(py::init<int, const std::string &>(),
             py::arg("value"),
             py::arg("name") = "")
        .def(py::init<int64_t, const std::string &>(),
             py::arg("value"),
             py::arg("name") = "")
        .def(py::init<double, const std::string &>(),
             py::arg("value"),
             py::arg("name") = "")
        .def(py::init<size_t, const std::string &>(),
             py::arg("value"),
             py::arg("name") = "")
        .def("get_value_kind", &Scalar::GetScalarValueKind)
        .def("has_constant_value", &Scalar::HasConstantValue)
        .def("to_int", &Scalar::GetInt64Value)
        .def("__int__", &Scalar::GetInt64Value)
        .def("__index__", &Scalar::GetInt64Value)
        .def("__repr__", [](const Scalar &s) {
            std::ostringstream os;
            s.Print(os);
            return os.str();
        })
        // Methods for scalar.py compatibility
        .def("IsImmediate", &Scalar::HasConstantValue)
        .def("IsSymbol", [](const Scalar &s) {
            return s.GetScalarValueKind() == ScalarValueKind::Symbolic && !s.HasConstantValue();
        })
        .def("IsExpression", [](const Scalar &s) {
            // An expression is a symbolic scalar that is not a simple symbol
            // For now, we consider all symbolic scalars as potential expressions
            return s.GetScalarValueKind() == ScalarValueKind::Symbolic;
        })
        .def("ConcreteValid", &Scalar::HasConstantValue)
        .def("Concrete", &Scalar::GetInt64Value)
        .def("AsIntermediateVariable", [](Scalar &s) {
            // This is a no-op for now, as we don't have intermediate variable tracking
            // The name is already set during construction
        })
        .def("Dump", [](const Scalar &s) {
            std::ostringstream os;
            s.Print(os);
            return os.str();
        })
        .def("__add__", [](const Scalar &self, py::object other) {
            int64_t lhs = self.GetInt64Value();
            int64_t rhs = 0;
            if (py::isinstance<py::int_>(other)) {
                rhs = other.cast<int64_t>();
            } else if (py::isinstance<Scalar>(other)) {
                rhs = other.cast<Scalar>().GetInt64Value();
            } else {
                throw py::type_error("Scalar can only be added with int or Scalar");
            }
            return Scalar(lhs + rhs);
        })
        .def("__radd__", [](const Scalar &self, py::object other) {
            int64_t rhs = self.GetInt64Value();
            int64_t lhs = 0;
            if (py::isinstance<py::int_>(other)) {
                lhs = other.cast<int64_t>();
            } else if (py::isinstance<Scalar>(other)) {
                lhs = other.cast<Scalar>().GetInt64Value();
            } else {
                throw py::type_error("Scalar can only be added with int or Scalar");
            }
            return Scalar(lhs + rhs);
        })
        .def("__sub__", [](const Scalar &self, py::object other) {
            int64_t lhs = self.GetInt64Value();
            int64_t rhs = 0;
            if (py::isinstance<py::int_>(other)) {
                rhs = other.cast<int64_t>();
            } else if (py::isinstance<Scalar>(other)) {
                rhs = other.cast<Scalar>().GetInt64Value();
            } else {
                throw py::type_error("Scalar can only be subtracted with int or Scalar");
            }
            return Scalar(lhs - rhs);
        })
        .def("__rsub__", [](const Scalar &self, py::object other) {
            int64_t rhs = self.GetInt64Value();
            int64_t lhs = 0;
            if (py::isinstance<py::int_>(other)) {
                lhs = other.cast<int64_t>();
            } else if (py::isinstance<Scalar>(other)) {
                lhs = other.cast<Scalar>().GetInt64Value();
            } else {
                throw py::type_error("Scalar can only be subtracted with int or Scalar");
            }
            return Scalar(lhs - rhs);
        })
        .def("__floordiv__", [](const Scalar &self, py::object other) {
            // Handle constant values
            if (self.HasConstantValue()) {
                int64_t lhs = self.GetInt64Value();
                int64_t rhs = 0;
                if (py::isinstance<py::int_>(other)) {
                    rhs = other.cast<int64_t>();
                } else if (py::isinstance<Scalar>(other)) {
                    const Scalar &other_scalar = other.cast<Scalar>();
                    if (other_scalar.HasConstantValue()) {
                        rhs = other_scalar.GetInt64Value();
                    } else {
                        // Symbolic division: create new symbolic scalar
                        std::ostringstream os;
                        self.Print(os);
                        os << " // ";
                        other_scalar.Print(os);
                        return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
                    }
                } else {
                    throw py::type_error("Scalar can only be floor-divided by int or Scalar");
                }
                if (rhs == 0) {
                    throw py::value_error("Division by zero");
                }
                return Scalar(lhs / rhs);
            } else {
                // Symbolic division: create new symbolic scalar
                std::ostringstream os;
                self.Print(os);
                os << " // ";
                if (py::isinstance<py::int_>(other)) {
                    os << other.cast<int64_t>();
                } else if (py::isinstance<Scalar>(other)) {
                    other.cast<Scalar>().Print(os);
                } else {
                    throw py::type_error("Scalar can only be floor-divided by int or Scalar");
                }
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("__rfloordiv__", [](const Scalar &self, py::object other) {
            // Handle constant values
            if (self.HasConstantValue()) {
                int64_t rhs = self.GetInt64Value();
                int64_t lhs = 0;
                if (py::isinstance<py::int_>(other)) {
                    lhs = other.cast<int64_t>();
                } else if (py::isinstance<Scalar>(other)) {
                    const Scalar &other_scalar = other.cast<Scalar>();
                    if (other_scalar.HasConstantValue()) {
                        lhs = other_scalar.GetInt64Value();
                    } else {
                        // Symbolic division: create new symbolic scalar
                        std::ostringstream os;
                        other_scalar.Print(os);
                        os << " // ";
                        self.Print(os);
                        return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
                    }
                } else {
                    throw py::type_error("Scalar can only be floor-divided by int or Scalar");
                }
                if (rhs == 0) {
                    throw py::value_error("Division by zero");
                }
                return Scalar(lhs / rhs);
            } else {
                // Symbolic division: create new symbolic scalar
                std::ostringstream os;
                if (py::isinstance<py::int_>(other)) {
                    os << other.cast<int64_t>();
                } else if (py::isinstance<Scalar>(other)) {
                    other.cast<Scalar>().Print(os);
                } else {
                    throw py::type_error("Scalar can only be floor-divided by int or Scalar");
                }
                os << " // ";
                self.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("__mul__", [](const Scalar &self, py::object other) {
            // Handle constant values
            if (self.HasConstantValue()) {
                int64_t lhs = self.GetInt64Value();
                int64_t rhs = 0;
                if (py::isinstance<py::int_>(other)) {
                    rhs = other.cast<int64_t>();
                } else if (py::isinstance<Scalar>(other)) {
                    const Scalar &other_scalar = other.cast<Scalar>();
                    if (other_scalar.HasConstantValue()) {
                        rhs = other_scalar.GetInt64Value();
                    } else {
                        // Symbolic multiplication: create new symbolic scalar
                        std::ostringstream os;
                        self.Print(os);
                        os << " * ";
                        other_scalar.Print(os);
                        return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
                    }
                } else {
                    throw py::type_error("Scalar can only be multiplied by int or Scalar");
                }
                return Scalar(lhs * rhs);
            } else {
                // Symbolic multiplication: create new symbolic scalar
                std::ostringstream os;
                self.Print(os);
                os << " * ";
                if (py::isinstance<py::int_>(other)) {
                    os << other.cast<int64_t>();
                } else if (py::isinstance<Scalar>(other)) {
                    other.cast<Scalar>().Print(os);
                } else {
                    throw py::type_error("Scalar can only be multiplied by int or Scalar");
                }
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("__rmul__", [](const Scalar &self, py::object other) {
            // Handle constant values
            if (self.HasConstantValue()) {
                int64_t rhs = self.GetInt64Value();
                int64_t lhs = 0;
                if (py::isinstance<py::int_>(other)) {
                    lhs = other.cast<int64_t>();
                } else if (py::isinstance<Scalar>(other)) {
                    const Scalar &other_scalar = other.cast<Scalar>();
                    if (other_scalar.HasConstantValue()) {
                        lhs = other_scalar.GetInt64Value();
                    } else {
                        // Symbolic multiplication: create new symbolic scalar
                        std::ostringstream os;
                        other_scalar.Print(os);
                        os << " * ";
                        self.Print(os);
                        return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
                    }
                } else {
                    throw py::type_error("Scalar can only be multiplied by int or Scalar");
                }
                return Scalar(lhs * rhs);
            } else {
                // Symbolic multiplication: create new symbolic scalar
                std::ostringstream os;
                if (py::isinstance<py::int_>(other)) {
                    os << other.cast<int64_t>();
                } else if (py::isinstance<Scalar>(other)) {
                    other.cast<Scalar>().Print(os);
                } else {
                    throw py::type_error("Scalar can only be multiplied by int or Scalar");
                }
                os << " * ";
                self.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        // Methods for scalar.py compatibility - binary operations
        .def("Add", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Add: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(self.GetInt64Value() + other_ref.GetInt64Value());
            } else {
                std::ostringstream os;
                self.Print(os);
                os << " + ";
                other_ref.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("RAdd", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("RAdd: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(other_ref.GetInt64Value() + self.GetInt64Value());
            } else {
                std::ostringstream os;
                other_ref.Print(os);
                os << " + ";
                self.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("Sub", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Sub: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(self.GetInt64Value() - other_ref.GetInt64Value());
            } else {
                std::ostringstream os;
                self.Print(os);
                os << " - ";
                other_ref.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("RSub", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("RSub: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(other_ref.GetInt64Value() - self.GetInt64Value());
            } else {
                std::ostringstream os;
                other_ref.Print(os);
                os << " - ";
                self.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("Mul", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Mul: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(self.GetInt64Value() * other_ref.GetInt64Value());
            } else {
                std::ostringstream os;
                self.Print(os);
                os << " * ";
                other_ref.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("RMul", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("RMul: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(other_ref.GetInt64Value() * self.GetInt64Value());
            } else {
                std::ostringstream os;
                other_ref.Print(os);
                os << " * ";
                self.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("Div", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Div: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                int64_t rhs = other_ref.GetInt64Value();
                if (rhs == 0) {
                    throw std::runtime_error("Division by zero");
                }
                return Scalar(self.GetInt64Value() / rhs);
            } else {
                std::ostringstream os;
                self.Print(os);
                os << " / ";
                other_ref.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("RDiv", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("RDiv: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                int64_t lhs = other_ref.GetInt64Value();
                int64_t rhs = self.GetInt64Value();
                if (rhs == 0) {
                    throw std::runtime_error("Division by zero");
                }
                return Scalar(lhs / rhs);
            } else {
                std::ostringstream os;
                other_ref.Print(os);
                os << " / ";
                self.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("RMod", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("RMod: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                int64_t lhs = other_ref.GetInt64Value();
                int64_t rhs = self.GetInt64Value();
                if (rhs == 0) {
                    throw std::runtime_error("Modulo by zero");
                }
                return Scalar(lhs % rhs);
            } else {
                std::ostringstream os;
                other_ref.Print(os);
                os << " % ";
                self.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        // Comparison operations
        .def("Eq", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Eq: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(self.GetInt64Value() == other_ref.GetInt64Value() ? 1 : 0);
            } else {
                std::ostringstream os;
                self.Print(os);
                os << " == ";
                other_ref.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("Ne", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Ne: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(self.GetInt64Value() != other_ref.GetInt64Value() ? 1 : 0);
            } else {
                std::ostringstream os;
                self.Print(os);
                os << " != ";
                other_ref.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("Lt", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Lt: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(self.GetInt64Value() < other_ref.GetInt64Value() ? 1 : 0);
            } else {
                std::ostringstream os;
                self.Print(os);
                os << " < ";
                other_ref.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("Le", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Le: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(self.GetInt64Value() <= other_ref.GetInt64Value() ? 1 : 0);
            } else {
                std::ostringstream os;
                self.Print(os);
                os << " <= ";
                other_ref.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("Gt", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Gt: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(self.GetInt64Value() > other_ref.GetInt64Value() ? 1 : 0);
            } else {
                std::ostringstream os;
                self.Print(os);
                os << " > ";
                other_ref.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("Ge", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Ge: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(self.GetInt64Value() >= other_ref.GetInt64Value() ? 1 : 0);
            } else {
                std::ostringstream os;
                self.Print(os);
                os << " >= ";
                other_ref.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        // Unary operations
        .def("Neg", [](const Scalar &self) {
            if (self.HasConstantValue()) {
                return Scalar(-self.GetInt64Value());
            } else {
                std::ostringstream os;
                os << "-";
                self.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("Pos", [](const Scalar &self) {
            // Positive is a no-op
            return self;
        })
        .def("Not", [](const Scalar &self) {
            if (self.HasConstantValue()) {
                return Scalar(self.GetInt64Value() == 0 ? 1 : 0);
            } else {
                std::ostringstream os;
                os << "!";
                self.Print(os);
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        // Min/Max operations
        .def("Min", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Min: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(std::min(self.GetInt64Value(), other_ref.GetInt64Value()));
            } else {
                std::ostringstream os;
                os << "min(";
                self.Print(os);
                os << ", ";
                other_ref.Print(os);
                os << ")";
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        })
        .def("Max", [](const Scalar &self, py::object other) {
            Scalar other_scalar;
            if (py::isinstance<py::int_>(other)) {
                other_scalar = Scalar(other.cast<int64_t>());
            } else if (py::isinstance<Scalar>(other)) {
                other_scalar = other.cast<Scalar>();
            } else {
                throw py::type_error("Max: argument must be int or Scalar");
            }
            const Scalar &other_ref = other_scalar;
            if (self.HasConstantValue() && other_ref.HasConstantValue()) {
                return Scalar(std::max(self.GetInt64Value(), other_ref.GetInt64Value()));
            } else {
                std::ostringstream os;
                os << "max(";
                self.Print(os);
                os << ", ";
                other_ref.Print(os);
                os << ")";
                return Scalar(DataType::INT64, os.str(), ScalarValueKind::Symbolic);
            }
        });
}

// Helper function to expose Value class (base class for Tensor, Scalar, etc.).
static void BindValue(py::module_ &m) {
    py::class_<Value, std::shared_ptr<Value>>(m, "Value")
        .def("GetValueKind", &Value::GetValueKind)
        .def("GetDataType", &Value::GetDataType)
        .def("GetSSAName", &Value::GetSSAName)
        .def("Print", [](const Value &v) {
            std::ostringstream os;
            v.Print(os, 0);
            return os.str();
        });
}

// Main type bindings function
void BindTypeBindings(py::module_ &m) {
    // DataType enum (Python) <-> DataType (C++).
    BindDataType(m);

    // TileOpFormat enum (Python) <-> TileOpFormat (C++).
    BindTileOpFormat(m);

    // Value class binding (base class for Tensor, Scalar, etc.).
    // MUST be bound before Tensor and Scalar since they inherit from Value.
    BindValue(m);

    // Scalar type used for symbolic and constant scalar values.
    BindScalar(m);

    // CastMode enum binding.
    py::enum_<CastMode>(m, "CastMode")
        .value("CAST_NONE", CastMode::CAST_NONE)
        .value("CAST_RINT", CastMode::CAST_RINT)
        .value("CAST_ROUND", CastMode::CAST_ROUND)
        .value("CAST_FLOOR", CastMode::CAST_FLOOR)
        .value("CAST_CEIL", CastMode::CAST_CEIL)
        .value("CAST_TRUNC", CastMode::CAST_TRUNC)
        .value("CAST_ODD", CastMode::CAST_ODD)
        .export_values();

    // ReLuType enum binding.
    py::enum_<ReLuType>(m, "ReLuType")
        .value("NoReLu", ReLuType::NoReLu)
        .value("ReLu", ReLuType::ReLu)
        .export_values();

    // ReduceKind enum binding.
    py::enum_<ReduceKind>(m, "ReduceKind")
        .value("RowSum", ReduceKind::RowSum)
        .value("RowMax", ReduceKind::RowMax)
        .export_values();
}

}  // namespace pto

