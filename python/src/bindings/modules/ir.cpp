/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "../bindings.h"
#include "core/any_cast.h"
#include "core/common.h"
#include "core/error.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/reflection/field_visitor.h"
#include "ir/scalar_expr.h"
#include "ir/serialization/deserializer.h"
#include "ir/serialization/serializer.h"
#include "ir/stmt.h"
#include "ir/transform/printer.h"
#include "ir/transform/structural_comparison.h"
#include "ir/type.h"

namespace py = pybind11;

namespace pypto {

using namespace pypto::ir;  // NOLINT(build/namespaces)

template <typename T>
bool TryConvertAnyToPy(const std::any &value, py::object &out) {
  if (value.type() != typeid(T)) {
    return false;
  }
  out = py::cast(AnyCastRef<T>(value, "converting to Python"));
  return true;
}

template <typename... Ts>
py::object AnyToPyObject(const std::any &value, const std::string &key) {
  py::object out;
  if ((TryConvertAnyToPy<Ts>(value, out) || ...)) {
    return out;
  }
  throw pypto::ir::TypeError("Attribute '" + key + "' has unsupported type");
}

template <typename ClassType, typename PyClassType, typename FieldDesc>
void BindField(PyClassType &pyClass, const FieldDesc &desc) {
  pyClass.def_readonly(desc.name, desc.field_ptr);
}

template <typename ClassType, typename PyClassType, typename DescTuple, std::size_t... Is>
void BindFieldsImpl(PyClassType &pyClass, const DescTuple &descriptors, std::index_sequence<Is...>) {
  (BindField<ClassType>(pyClass, std::get<Is>(descriptors)), ...);
}

template <typename ClassType, typename PyClassType>
void BindFields(PyClassType &pyClass) {
  constexpr auto descriptors = ClassType::GetFieldDescriptors();
  constexpr auto numFields = std::tuple_size_v<decltype(descriptors)>;
  BindFieldsImpl<ClassType>(pyClass, descriptors, std::make_index_sequence<numFields>{});
}

std::vector<std::pair<std::string, std::any>> ConvertKwargsDict(const py::dict &kwargsDict) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  for (auto item : kwargsDict) {
    std::string key = py::cast<std::string>(item.first);
    py::handle val = item.second;

    if (py::isinstance<DataType>(val)) {
      kwargs.emplace_back(key, py::cast<DataType>(val));
    } else if (py::isinstance<py::bool_>(val)) {
      kwargs.emplace_back(key, py::cast<bool>(val));
    } else if (py::isinstance<py::int_>(val)) {
      kwargs.emplace_back(key, py::cast<int>(val));
    } else if (py::isinstance<py::str>(val)) {
      kwargs.emplace_back(key, py::cast<std::string>(val));
    } else if (py::isinstance<py::float_>(val)) {
      kwargs.emplace_back(key, py::cast<double>(val));
    } else if (py::isinstance<PipeType>(val)) {
      kwargs.emplace_back(key, static_cast<int>(py::cast<PipeType>(val)));
    } else if (py::isinstance<CoreType>(val)) {
      kwargs.emplace_back(key, static_cast<int>(py::cast<CoreType>(val)));
    } else {
      throw pypto::ir::TypeError("Unsupported kwarg type for key: " + key);
    }
  }
  return kwargs;
}

void BindIR(py::module_ &m) {
  py::module_ ir = m.def_submodule("ir", "PyPTO IR (Intermediate Representation) module");

  // Span
  py::class_<Span>(ir, "Span")
      .def(py::init<std::string, int, int, int, int>(), py::arg("filename"), py::arg("begin_line"),
           py::arg("begin_column"), py::arg("end_line") = -1, py::arg("end_column") = -1)
      .def("to_string", &Span::to_string)
      .def("is_valid", &Span::is_valid)
      .def_static("unknown", &Span::unknown)
      .def("__repr__", &Span::to_string)
      .def("__str__", &Span::to_string)
      .def_readonly("filename", &Span::filename_)
      .def_readonly("begin_line", &Span::beginLine_)
      .def_readonly("begin_column", &Span::beginColumn_)
      .def_readonly("end_line", &Span::endLine_)
      .def_readonly("end_column", &Span::endColumn_);

  // Op
  py::class_<Op, std::shared_ptr<Op>>(ir, "Op")
      .def(py::init<std::string>(), py::arg("name"))
      .def_readonly("name", &Op::name_)
      .def("has_attr", &Op::HasAttr, py::arg("key"))
      .def("get_attr_keys", &Op::GetAttrKeys)
      .def_property_readonly("pipe", [](const Op &self) -> std::optional<PipeType> { return self.GetPipe(); });

  // GlobalVar
  py::class_<GlobalVar, Op, std::shared_ptr<GlobalVar>>(ir, "GlobalVar")
      .def(py::init<std::string>(), py::arg("name"));

  // Type hierarchy
  auto typeClass = py::class_<Type, std::shared_ptr<Type>>(ir, "Type");
  BindFields<Type>(typeClass);
  typeClass.def("__str__", [](const TypePtr &self) { return PythonPrint(self, "pl"); });
  typeClass.def("__eq__", [](const TypePtr &self, const TypePtr &other) { return structural_equal(self, other); });

  auto unknownTypeClass = py::class_<UnknownType, Type, std::shared_ptr<UnknownType>>(ir, "UnknownType");
  unknownTypeClass.def(py::init<>());
  unknownTypeClass.def_static("get", []() { return GetUnknownType(); });
  BindFields<UnknownType>(unknownTypeClass);

  auto scalarTypeClass = py::class_<ScalarType, Type, std::shared_ptr<ScalarType>>(ir, "ScalarType");
  scalarTypeClass.def(py::init<DataType>(), py::arg("dtype"));
  BindFields<ScalarType>(scalarTypeClass);

  // IRNode
  auto irnodeClass = py::class_<IRNode, std::shared_ptr<IRNode>>(ir, "IRNode");
  BindFields<IRNode>(irnodeClass);
  irnodeClass
      .def("same_as", [](const IRNodePtr &self, const IRNodePtr &other) { return self == other; }, py::arg("other"))
      .def("__str__", [](const IRNodePtr &self) { return PythonPrint(self, "pl"); })
      .def("as_python", [](const IRNodePtr &self, const std::string &prefix) { return PythonPrint(self, prefix); },
           py::arg("prefix") = "pl");

  // Expr
  auto exprClass = py::class_<Expr, IRNode, std::shared_ptr<Expr>>(ir, "Expr");
  BindFields<Expr>(exprClass);

  // ShapedType
  auto shapedTypeClass = py::class_<ShapedType, Type, std::shared_ptr<ShapedType>>(ir, "ShapedType");
  BindFields<ShapedType>(shapedTypeClass);
  shapedTypeClass.def("shares_memref_with",
      [](const ShapedTypePtr &self, const ShapedTypePtr &other) {
        if (!self->memref_.has_value() || !other->memref_.has_value()) return false;
        return self->memref_.value().get() == other->memref_.value().get();
      }, py::arg("other"));

  // TensorType
  auto tensorTypeClass = py::class_<TensorType, ShapedType, std::shared_ptr<TensorType>>(ir, "TensorType");
  tensorTypeClass.def(py::init<const std::vector<ExprPtr> &, DataType, std::optional<MemRefPtr>>(),
      py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none());
  tensorTypeClass.def(py::init<const std::vector<int64_t> &, DataType, std::optional<MemRefPtr>>(),
      py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none());
  BindFields<TensorType>(tensorTypeClass);

  // TileType
  auto tileTypeClass = py::class_<TileType, ShapedType, std::shared_ptr<TileType>>(ir, "TileType");
  tileTypeClass.def(py::init<const std::vector<ExprPtr> &, DataType,
                                std::optional<MemRefPtr>, std::optional<TileView>>(),
      py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(), py::arg("tile_view") = py::none());
  tileTypeClass.def(py::init<const std::vector<int64_t> &, DataType,
                                std::optional<MemRefPtr>, std::optional<TileView>>(),
      py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(), py::arg("tile_view") = py::none());
  BindFields<TileType>(tileTypeClass);

  // TupleType
  auto tupleTypeClass = py::class_<TupleType, Type, std::shared_ptr<TupleType>>(ir, "TupleType");
  tupleTypeClass.def(py::init<const std::vector<TypePtr> &>(), py::arg("types"));
  BindFields<TupleType>(tupleTypeClass);

  // Enums
  py::enum_<MemorySpace>(ir, "MemorySpace")
      .value("DDR", MemorySpace::DDR)
      .value("UB", MemorySpace::UB)
      .value("L1", MemorySpace::L1)
      .value("L0A", MemorySpace::L0A)
      .value("L0B", MemorySpace::L0B)
      .value("L0C", MemorySpace::L0C)
      .export_values();

  py::enum_<PipeType>(ir, "PipeType", py::arithmetic())
      .value("MTE1", PipeType::MTE1)
      .value("MTE2", PipeType::MTE2)
      .value("MTE3", PipeType::MTE3)
      .value("M", PipeType::M)
      .value("V", PipeType::V)
      .value("S", PipeType::S)
      .value("FIX", PipeType::FIX)
      .value("ALL", PipeType::ALL)
      .export_values();

  py::enum_<CoreType>(ir, "CoreType", py::arithmetic())
      .value("VECTOR", CoreType::VECTOR)
      .value("CUBE", CoreType::CUBE)
      .export_values();

  // TileView
  py::class_<TileView>(ir, "TileView")
      .def(py::init<>())
      .def(py::init<const std::vector<ExprPtr> &, const std::vector<ExprPtr> &, ExprPtr>(),
           py::arg("valid_shape"), py::arg("stride"), py::arg("start_offset"))
      .def_readwrite("valid_shape", &TileView::validShape)
      .def_readwrite("stride", &TileView::stride)
      .def_readwrite("start_offset", &TileView::startOffset);

  ir.attr("DYNAMIC_DIM") = kDynamicDim;

  // OpRegistry
  ir.def("create_op_call",
      [](const std::string &opName, const std::vector<ExprPtr> &args, const Span &span) {
        return OpRegistry::GetInstance().Create(opName, args, span);
      }, py::arg("op_name"), py::arg("args"), py::arg("span"));
  ir.def("create_op_call",
      [](const std::string &opName, const std::vector<ExprPtr> &args, const py::dict &kwargsDict,
         const Span &span) {
        auto kwargs = ConvertKwargsDict(kwargsDict);
        return OpRegistry::GetInstance().Create(opName, args, kwargs, span);
      }, py::arg("op_name"), py::arg("args"), py::arg("kwargs"), py::arg("span"));
  ir.def("is_op_registered",
      [](const std::string &opName) { return OpRegistry::GetInstance().IsRegistered(opName); },
      py::arg("op_name"));
  ir.def("get_op",
      [](const std::string &opName) { return OpRegistry::GetInstance().GetOp(opName); },
      py::arg("op_name"));

  // Var
  auto varClass = py::class_<Var, Expr, std::shared_ptr<Var>>(ir, "Var");
  varClass.def(py::init<const std::string &, const TypePtr &, const Span &>(),
                py::arg("name"), py::arg("type"), py::arg("span"));
  BindFields<Var>(varClass);

  // IterArg
  auto iterargClass = py::class_<IterArg, Var, std::shared_ptr<IterArg>>(ir, "IterArg");
  iterargClass.def(py::init<const std::string &, const TypePtr &, const ExprPtr &, const Span &>(),
                    py::arg("name"), py::arg("type"), py::arg("initValue"), py::arg("span"));
  BindFields<IterArg>(iterargClass);

  // MemRef
  auto memrefClass = py::class_<MemRef, Var, std::shared_ptr<MemRef>>(ir, "MemRef");
  memrefClass
      .def(py::init<MemorySpace, ExprPtr, uint64_t, uint64_t, Span>(), py::arg("memory_space"),
           py::arg("addr"), py::arg("size"), py::arg("id"), py::arg("span") = Span::unknown())
      .def_readwrite("memory_space_", &MemRef::memorySpace_)
      .def_readwrite("addr_", &MemRef::addr_)
      .def_readwrite("size_", &MemRef::size_)
      .def_readwrite("id_", &MemRef::id_);

  // ConstInt
  auto constintClass = py::class_<ConstInt, Expr, std::shared_ptr<ConstInt>>(ir, "ConstInt");
  constintClass.def(py::init<int64_t, DataType, const Span &>(),
      py::arg("value"), py::arg("dtype"), py::arg("span"));
  BindFields<ConstInt>(constintClass);
  constintClass.def_property_readonly("dtype", &ConstInt::dtype);

  // ConstFloat
  auto constfloatClass = py::class_<ConstFloat, Expr, std::shared_ptr<ConstFloat>>(ir, "ConstFloat");
  constfloatClass.def(py::init<double, DataType, const Span &>(),
      py::arg("value"), py::arg("dtype"), py::arg("span"));
  BindFields<ConstFloat>(constfloatClass);
  constfloatClass.def_property_readonly("dtype", &ConstFloat::dtype);

  // ConstBool
  auto constboolClass = py::class_<ConstBool, Expr, std::shared_ptr<ConstBool>>(ir, "ConstBool");
  constboolClass.def(py::init<bool, const Span &>(), py::arg("value"), py::arg("span"));
  BindFields<ConstBool>(constboolClass);
  constboolClass.def_property_readonly("dtype", &ConstBool::dtype);

  // Call
  auto callClass = py::class_<Call, Expr, std::shared_ptr<Call>>(ir, "Call");
  callClass.def(py::init<const OpPtr &, const std::vector<ExprPtr> &, const Span &>(),
                 py::arg("op"), py::arg("args"), py::arg("span"));
  callClass.def(py::init<const OpPtr &, const std::vector<ExprPtr> &, const TypePtr &, const Span &>(),
                 py::arg("op"), py::arg("args"), py::arg("type"), py::arg("span"));
  callClass.def(py::init([](const OpPtr &op, const std::vector<ExprPtr> &args, const py::dict &kwargsDict,
                              const Span &span) {
    auto kwargs = ConvertKwargsDict(kwargsDict);
    return std::make_shared<Call>(op, args, kwargs, span);
  }), py::arg("op"), py::arg("args"), py::arg("kwargs"), py::arg("span"));
  callClass.def(py::init([](const OpPtr &op, const std::vector<ExprPtr> &args, const py::dict &kwargsDict,
                              const TypePtr &type, const Span &span) {
    auto kwargs = ConvertKwargsDict(kwargsDict);
    return std::make_shared<Call>(op, args, kwargs, type, span);
  }), py::arg("op"), py::arg("args"), py::arg("kwargs"), py::arg("type"), py::arg("span"));
  BindFields<Call>(callClass);
  callClass.def_property_readonly("kwargs", [](const CallPtr &self) {
    py::dict result;
    for (const auto &[key, value] : self->kwargs_) {
      if (value.type() == typeid(int)) {
        result[key.c_str()] = AnyCast<int>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(bool)) {
        result[key.c_str()] = AnyCast<bool>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(std::string)) {
        result[key.c_str()] = AnyCast<std::string>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(double)) {
        result[key.c_str()] = AnyCast<double>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(float)) {
        result[key.c_str()] = AnyCast<float>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(DataType)) {
        result[key.c_str()] = AnyCast<DataType>(value, "converting to Python: " + key);
      }
    }
    return result;
  });

  // MakeTuple
  auto makeTupleClass = py::class_<MakeTuple, Expr, std::shared_ptr<MakeTuple>>(ir, "MakeTuple");
  makeTupleClass.def(py::init<const std::vector<ExprPtr> &, const Span &>(),
                       py::arg("elements"), py::arg("span"));
  BindFields<MakeTuple>(makeTupleClass);

  // TupleGetItemExpr
  auto tupleGetItemClass = py::class_<TupleGetItemExpr, Expr, std::shared_ptr<TupleGetItemExpr>>(ir, "TupleGetItemExpr");
  tupleGetItemClass.def(py::init<const ExprPtr &, int, const Span &>(),
                           py::arg("tuple"), py::arg("index"), py::arg("span"));
  BindFields<TupleGetItemExpr>(tupleGetItemClass);

  // BinaryExpr
  auto binaryexprClass = py::class_<BinaryExpr, Expr, std::shared_ptr<BinaryExpr>>(ir, "BinaryExpr");
  BindFields<BinaryExpr>(binaryexprClass);

  // UnaryExpr
  auto unaryexprClass = py::class_<UnaryExpr, Expr, std::shared_ptr<UnaryExpr>>(ir, "UnaryExpr");
  BindFields<UnaryExpr>(unaryexprClass);

#define BIND_BINARY_EXPR(OpName, Description)                                                  \
  py::class_<OpName, BinaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description)            \
      .def(py::init<const ExprPtr &, const ExprPtr &, DataType, const Span &>(),               \
           py::arg("left"), py::arg("right"), py::arg("dtype"), py::arg("span"));

  BIND_BINARY_EXPR(Add, "Addition expression")
  BIND_BINARY_EXPR(Sub, "Subtraction expression")
  BIND_BINARY_EXPR(Mul, "Multiplication expression")
  BIND_BINARY_EXPR(FloorDiv, "Floor division expression")
  BIND_BINARY_EXPR(FloorMod, "Floor modulo expression")
  BIND_BINARY_EXPR(FloatDiv, "Float division expression")
  BIND_BINARY_EXPR(Min, "Minimum expression")
  BIND_BINARY_EXPR(Max, "Maximum expression")
  BIND_BINARY_EXPR(Pow, "Power expression")
  BIND_BINARY_EXPR(Eq, "Equality expression")
  BIND_BINARY_EXPR(Ne, "Inequality expression")
  BIND_BINARY_EXPR(Lt, "Less than expression")
  BIND_BINARY_EXPR(Le, "Less than or equal expression")
  BIND_BINARY_EXPR(Gt, "Greater than expression")
  BIND_BINARY_EXPR(Ge, "Greater than or equal expression")
  BIND_BINARY_EXPR(And, "Logical and expression")
  BIND_BINARY_EXPR(Or, "Logical or expression")
  BIND_BINARY_EXPR(Xor, "Logical xor expression")
  BIND_BINARY_EXPR(BitAnd, "Bitwise and expression")
  BIND_BINARY_EXPR(BitOr, "Bitwise or expression")
  BIND_BINARY_EXPR(BitXor, "Bitwise xor expression")
  BIND_BINARY_EXPR(BitShiftLeft, "Bitwise left shift expression")
  BIND_BINARY_EXPR(BitShiftRight, "Bitwise right shift expression")

#undef BIND_BINARY_EXPR

#define BIND_UNARY_EXPR(OpName, Description)                                                        \
  py::class_<OpName, UnaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description)                  \
      .def(py::init<const ExprPtr &, DataType, const Span &>(),                                     \
           py::arg("operand"), py::arg("dtype"), py::arg("span"));

  BIND_UNARY_EXPR(Abs, "Absolute value expression")
  BIND_UNARY_EXPR(Neg, "Negation expression")
  BIND_UNARY_EXPR(Not, "Logical not expression")
  BIND_UNARY_EXPR(BitNot, "Bitwise not expression")
  BIND_UNARY_EXPR(Cast, "Cast expression")

#undef BIND_UNARY_EXPR

  // Structural hash and equality
  ir.def("structural_hash", static_cast<uint64_t (*)(const IRNodePtr &, bool)>(&structural_hash),
         py::arg("node"), py::arg("enable_auto_mapping") = false);
  ir.def("structural_hash", static_cast<uint64_t (*)(const TypePtr &, bool)>(&structural_hash),
         py::arg("type"), py::arg("enable_auto_mapping") = false);
  ir.def("structural_equal",
         static_cast<bool (*)(const IRNodePtr &, const IRNodePtr &, bool)>(&structural_equal),
         py::arg("lhs"), py::arg("rhs"), py::arg("enable_auto_mapping") = false);
  ir.def("structural_equal",
         static_cast<bool (*)(const TypePtr &, const TypePtr &, bool)>(&structural_equal),
         py::arg("lhs"), py::arg("rhs"), py::arg("enable_auto_mapping") = false);
  ir.def("assert_structural_equal",
         static_cast<void (*)(const IRNodePtr &, const IRNodePtr &, bool)>(&assert_structural_equal),
         py::arg("lhs"), py::arg("rhs"), py::arg("enable_auto_mapping") = false);
  ir.def("assert_structural_equal",
         static_cast<void (*)(const TypePtr &, const TypePtr &, bool)>(&assert_structural_equal),
         py::arg("lhs"), py::arg("rhs"), py::arg("enable_auto_mapping") = false);

  // Serialization
  ir.def("serialize", [](const IRNodePtr &node) {
    auto data = serialization::Serialize(node);
    return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
  }, py::arg("node"));
  ir.def("deserialize", [](const py::bytes &data) {
    std::string str = data;
    std::vector<uint8_t> vec(reinterpret_cast<const uint8_t *>(str.data()),
                             reinterpret_cast<const uint8_t *>(str.data()) + str.size());
    return serialization::Deserialize(vec);
  }, py::arg("data"));
  ir.def("serialize_to_file", &serialization::SerializeToFile, py::arg("node"), py::arg("path"));
  ir.def("deserialize_from_file", &serialization::DeserializeFromFile, py::arg("path"));

  // Statements
  auto stmtClass = py::class_<Stmt, IRNode, std::shared_ptr<Stmt>>(ir, "Stmt");
  BindFields<Stmt>(stmtClass);

  auto assignStmtClass = py::class_<AssignStmt, Stmt, std::shared_ptr<AssignStmt>>(ir, "AssignStmt");
  assignStmtClass.def(py::init<const VarPtr &, const ExprPtr &, const Span &>(),
                        py::arg("var"), py::arg("value"), py::arg("span"));
  BindFields<AssignStmt>(assignStmtClass);

  auto ifStmtClass = py::class_<IfStmt, Stmt, std::shared_ptr<IfStmt>>(ir, "IfStmt");
  ifStmtClass.def(py::init<const ExprPtr &, const StmtPtr &, const std::optional<StmtPtr> &,
                             const std::vector<VarPtr> &, const Span &>(),
                    py::arg("condition"), py::arg("then_body"), py::arg("else_body") = py::none(),
                    py::arg("return_vars"), py::arg("span"));
  BindFields<IfStmt>(ifStmtClass);

  auto yieldStmtClass = py::class_<YieldStmt, Stmt, std::shared_ptr<YieldStmt>>(ir, "YieldStmt");
  yieldStmtClass.def(py::init<const std::vector<ExprPtr> &, const Span &>(),
                       py::arg("value"), py::arg("span"));
  yieldStmtClass.def(py::init<const Span &>(), py::arg("span"));
  BindFields<YieldStmt>(yieldStmtClass);

  auto returnStmtClass = py::class_<ReturnStmt, Stmt, std::shared_ptr<ReturnStmt>>(ir, "ReturnStmt");
  returnStmtClass.def(py::init<const std::vector<ExprPtr> &, const Span &>(),
                        py::arg("value"), py::arg("span"));
  returnStmtClass.def(py::init<const Span &>(), py::arg("span"));
  BindFields<ReturnStmt>(returnStmtClass);

  auto forStmtClass = py::class_<ForStmt, Stmt, std::shared_ptr<ForStmt>>(ir, "ForStmt");
  forStmtClass.def(
      py::init<const VarPtr &, const ExprPtr &, const ExprPtr &, const ExprPtr &, const std::vector<IterArgPtr> &,
               const StmtPtr &, const std::vector<VarPtr> &, const Span &>(),
      py::arg("loop_var"), py::arg("start"), py::arg("stop"), py::arg("step"), py::arg("iter_args"),
      py::arg("body"), py::arg("return_vars"), py::arg("span"));
  BindFields<ForStmt>(forStmtClass);

  auto seqStmtsClass = py::class_<SeqStmts, Stmt, std::shared_ptr<SeqStmts>>(ir, "SeqStmts");
  seqStmtsClass.def(py::init<const std::vector<StmtPtr> &, const Span &>(),
                      py::arg("stmts"), py::arg("span"));
  BindFields<SeqStmts>(seqStmtsClass);

  auto opStmtsClass = py::class_<OpStmts, Stmt, std::shared_ptr<OpStmts>>(ir, "OpStmts");
  opStmtsClass.def(py::init<const std::vector<StmtPtr> &, const Span &>(),
                     py::arg("stmts"), py::arg("span"));
  BindFields<OpStmts>(opStmtsClass);

  auto evalStmtClass = py::class_<EvalStmt, Stmt, std::shared_ptr<EvalStmt>>(ir, "EvalStmt");
  evalStmtClass.def(py::init<const ExprPtr &, const Span &>(), py::arg("expr"), py::arg("span"));
  BindFields<EvalStmt>(evalStmtClass);

  // FunctionType enum
  py::enum_<FunctionType>(ir, "FunctionType")
      .value("Opaque", FunctionType::OPAQUE)
      .value("Orchestration", FunctionType::ORCHESTRATION)
      .value("InCore", FunctionType::IN_CORE)
      .export_values();

  // Function
  auto functionClass = py::class_<Function, IRNode, std::shared_ptr<Function>>(ir, "Function");
  functionClass.def(py::init<const std::string &, const std::vector<VarPtr> &, const std::vector<TypePtr> &,
                              const StmtPtr &, const Span &, FunctionType>(),
                     py::arg("name"), py::arg("params"), py::arg("return_types"), py::arg("body"),
                     py::arg("span"), py::arg("type") = FunctionType::OPAQUE);
  BindFields<Function>(functionClass);

  // Program
  auto programClass = py::class_<Program, IRNode, std::shared_ptr<Program>>(ir, "Program");
  programClass.def(py::init<const std::vector<FunctionPtr> &, const std::string &, const Span &>(),
                    py::arg("functions"), py::arg("name"), py::arg("span"));
  programClass.def("get_function", &Program::GetFunction, py::arg("name"));
  programClass.def("get_global_var", &Program::GetGlobalVar, py::arg("name"));
  programClass.def_property_readonly("functions",
      [](const std::shared_ptr<const Program> &self) {
        py::dict result;
        for (const auto &[gvar, func] : self->functions_) {
          result[py::cast(gvar)] = py::cast(func);
        }
        return result;
      });
  programClass.def_readonly("name", &Program::name_);
  programClass.def_readonly("span", &Program::span_);

  // Python-style printer
  ir.def("python_print",
      [](const IRNodePtr &node, const std::string &prefix) { return PythonPrint(node, prefix); },
      py::arg("node"), py::arg("prefix") = "pl");
  ir.def("python_print_type",
      [](const TypePtr &type, const std::string &prefix) { return PythonPrint(type, prefix); },
      py::arg("type"), py::arg("prefix") = "pl");

  // Scalar expression operators
  ir.def("add", &MakeAdd, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("sub", &MakeSub, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("mul", &MakeMul, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("truediv", &MakeFloatDiv, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("floordiv", &MakeFloorDiv, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("mod", &MakeFloorMod, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("pow", &MakePow, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("eq", &MakeEq, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("ne", &MakeNe, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("lt", &MakeLt, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("le", &MakeLe, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("gt", &MakeGt, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("ge", &MakeGe, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("neg", &MakeNeg, py::arg("operand"), py::arg("span") = Span::unknown());
  ir.def("cast", &MakeCast, py::arg("operand"), py::arg("dtype"), py::arg("span") = Span::unknown());
  ir.def("bit_and", &MakeBitAnd, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("bit_or", &MakeBitOr, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("bit_xor", &MakeBitXor, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("bit_shift_left", &MakeBitShiftLeft, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("bit_shift_right", &MakeBitShiftRight, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown());
  ir.def("bit_not", &MakeBitNot, py::arg("operand"), py::arg("span") = Span::unknown());
}

} // namespace pypto
