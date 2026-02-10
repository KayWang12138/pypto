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

#include "bindings.h"
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
#include "tilefwk/data_type.h"

namespace py = pybind11;

namespace pypto {

using namespace pypto::ir;  // NOLINT(build/namespaces)
using FwkDataType = npu::tile_fwk::DataType;  // Framework DataType enum

// Convert framework DataType enum to IR DataType class
inline DataType ConvertDataType(FwkDataType fwk_dtype) {
  switch (fwk_dtype) {
    case FwkDataType::DT_BOOL: return DataType::BOOL;
    case FwkDataType::DT_INT4: return DataType::INT4;
    case FwkDataType::DT_INT8: return DataType::INT8;
    case FwkDataType::DT_INT16: return DataType::INT16;
    case FwkDataType::DT_INT32: return DataType::INT32;
    case FwkDataType::DT_INT64: return DataType::INT64;
    case FwkDataType::DT_UINT8: return DataType::UINT8;
    case FwkDataType::DT_UINT16: return DataType::UINT16;
    case FwkDataType::DT_UINT32: return DataType::UINT32;
    case FwkDataType::DT_UINT64: return DataType::UINT64;
    case FwkDataType::DT_FP8: return DataType::FP8;
    case FwkDataType::DT_FP8E4M3: return DataType::FP8E4M3FN;
    case FwkDataType::DT_FP8E5M2: return DataType::FP8E5M2;
    case FwkDataType::DT_FP16: return DataType::FP16;
    case FwkDataType::DT_FP32: return DataType::FP32;
    case FwkDataType::DT_BF16: return DataType::BF16;
    case FwkDataType::DT_HF4: return DataType::HF4;
    case FwkDataType::DT_HF8: return DataType::HF8;
    case FwkDataType::DT_DOUBLE: return DataType::FP32;  // Map DOUBLE to FP32 for now
    default:
      throw std::runtime_error("Unsupported DataType conversion");
  }
}

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
void BindField(PyClassType &py_class, const FieldDesc &desc) {
  py_class.def_readonly(desc.name, desc.field_ptr);
}

template <typename ClassType, typename PyClassType, typename DescTuple, std::size_t... Is>
void BindFieldsImpl(PyClassType &py_class, const DescTuple &descriptors, std::index_sequence<Is...>) {
  (BindField<ClassType>(py_class, std::get<Is>(descriptors)), ...);
}

template <typename ClassType, typename PyClassType>
void BindFields(PyClassType &py_class) {
  constexpr auto descriptors = ClassType::GetFieldDescriptors();
  constexpr auto num_fields = std::tuple_size_v<decltype(descriptors)>;
  BindFieldsImpl<ClassType>(py_class, descriptors, std::make_index_sequence<num_fields>{});
}

std::vector<std::pair<std::string, std::any>> ConvertKwargsDict(const py::dict &kwargs_dict) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  for (auto item : kwargs_dict) {
    std::string key = py::cast<std::string>(item.first);
    py::handle val = item.second;

    if (py::isinstance<FwkDataType>(val)) {
      // Convert framework DataType enum to IR DataType class
      kwargs.emplace_back(key, ConvertDataType(py::cast<FwkDataType>(val)));
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
      .def_readonly("begin_line", &Span::begin_line_)
      .def_readonly("begin_column", &Span::begin_column_)
      .def_readonly("end_line", &Span::end_line_)
      .def_readonly("end_column", &Span::end_column_);

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
  auto type_class = py::class_<Type, std::shared_ptr<Type>>(ir, "Type");
  BindFields<Type>(type_class);
  type_class.def("__str__", [](const TypePtr &self) { return PythonPrint(self, "pl"); });
  type_class.def("__eq__", [](const TypePtr &self, const TypePtr &other) { return structural_equal(self, other); });

  auto unknown_type_class = py::class_<UnknownType, Type, std::shared_ptr<UnknownType>>(ir, "UnknownType");
  unknown_type_class.def(py::init<>());
  unknown_type_class.def_static("get", []() { return GetUnknownType(); });
  BindFields<UnknownType>(unknown_type_class);

  auto scalar_type_class = py::class_<ScalarType, Type, std::shared_ptr<ScalarType>>(ir, "ScalarType");
  scalar_type_class.def(py::init([](FwkDataType dtype) {
    return std::make_shared<ScalarType>(ConvertDataType(dtype));
  }), py::arg("dtype"));
  BindFields<ScalarType>(scalar_type_class);

  // IRNode
  auto irnode_class = py::class_<IRNode, std::shared_ptr<IRNode>>(ir, "IRNode");
  BindFields<IRNode>(irnode_class);
  irnode_class
      .def("same_as", [](const IRNodePtr &self, const IRNodePtr &other) { return self == other; }, py::arg("other"))
      .def("__str__", [](const IRNodePtr &self) { return PythonPrint(self, "pl"); })
      .def("as_python", [](const IRNodePtr &self, const std::string &prefix) { return PythonPrint(self, prefix); },
           py::arg("prefix") = "pl");

  // Expr
  auto expr_class = py::class_<Expr, IRNode, std::shared_ptr<Expr>>(ir, "Expr");
  BindFields<Expr>(expr_class);

  // ShapedType
  auto shaped_type_class = py::class_<ShapedType, Type, std::shared_ptr<ShapedType>>(ir, "ShapedType");
  BindFields<ShapedType>(shaped_type_class);
  shaped_type_class.def("shares_memref_with",
      [](const ShapedTypePtr &self, const ShapedTypePtr &other) {
        if (!self->memref_.has_value() || !other->memref_.has_value()) return false;
        return self->memref_.value().get() == other->memref_.value().get();
      }, py::arg("other"));

  // TensorType
  auto tensor_type_class = py::class_<TensorType, ShapedType, std::shared_ptr<TensorType>>(ir, "TensorType");
  tensor_type_class.def(py::init([](const std::vector<ExprPtr> &shape, FwkDataType dtype, std::optional<MemRefPtr> memref) {
    return std::make_shared<TensorType>(shape, ConvertDataType(dtype), memref);
  }), py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none());
  tensor_type_class.def(py::init([](const std::vector<int64_t> &shape, FwkDataType dtype, std::optional<MemRefPtr> memref) {
    return std::make_shared<TensorType>(shape, ConvertDataType(dtype), memref);
  }), py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none());
  BindFields<TensorType>(tensor_type_class);

  // TileType
  auto tile_type_class = py::class_<TileType, ShapedType, std::shared_ptr<TileType>>(ir, "TileType");
  tile_type_class.def(py::init([](const std::vector<ExprPtr> &shape, FwkDataType dtype,
                                   std::optional<MemRefPtr> memref, std::optional<TileView> tile_view) {
    return std::make_shared<TileType>(shape, ConvertDataType(dtype), memref, tile_view);
  }), py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(), py::arg("tile_view") = py::none());
  tile_type_class.def(py::init([](const std::vector<int64_t> &shape, FwkDataType dtype,
                                   std::optional<MemRefPtr> memref, std::optional<TileView> tile_view) {
    return std::make_shared<TileType>(shape, ConvertDataType(dtype), memref, tile_view);
  }), py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(), py::arg("tile_view") = py::none());
  BindFields<TileType>(tile_type_class);

  // TupleType
  auto tuple_type_class = py::class_<TupleType, Type, std::shared_ptr<TupleType>>(ir, "TupleType");
  tuple_type_class.def(py::init<const std::vector<TypePtr> &>(), py::arg("types"));
  BindFields<TupleType>(tuple_type_class);
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
      .def_readwrite("valid_shape", &TileView::valid_shape)
      .def_readwrite("stride", &TileView::stride)
      .def_readwrite("start_offset", &TileView::start_offset);

  ir.attr("DYNAMIC_DIM") = kDynamicDim;

  // OpRegistry
  ir.def("create_op_call",
      [](const std::string &op_name, const std::vector<ExprPtr> &args, const Span &span) {
        return OpRegistry::GetInstance().Create(op_name, args, span);
      }, py::arg("op_name"), py::arg("args"), py::arg("span"));
  ir.def("create_op_call",
      [](const std::string &op_name, const std::vector<ExprPtr> &args, const py::dict &kwargs_dict,
         const Span &span) {
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        return OpRegistry::GetInstance().Create(op_name, args, kwargs, span);
      }, py::arg("op_name"), py::arg("args"), py::arg("kwargs"), py::arg("span"));
  ir.def("is_op_registered",
      [](const std::string &op_name) { return OpRegistry::GetInstance().IsRegistered(op_name); },
      py::arg("op_name"));
  ir.def("get_op",
      [](const std::string &op_name) { return OpRegistry::GetInstance().GetOp(op_name); },
      py::arg("op_name"));
  // Var
  auto var_class = py::class_<Var, Expr, std::shared_ptr<Var>>(ir, "Var");
  var_class.def(py::init<const std::string &, const TypePtr &, const Span &>(),
                py::arg("name"), py::arg("type"), py::arg("span"));
  BindFields<Var>(var_class);

  // IterArg
  auto iterarg_class = py::class_<IterArg, Var, std::shared_ptr<IterArg>>(ir, "IterArg");
  iterarg_class.def(py::init<const std::string &, const TypePtr &, const ExprPtr &, const Span &>(),
                    py::arg("name"), py::arg("type"), py::arg("initValue"), py::arg("span"));
  BindFields<IterArg>(iterarg_class);

  // MemRef
  auto memref_class = py::class_<MemRef, Var, std::shared_ptr<MemRef>>(ir, "MemRef");
  memref_class
      .def(py::init<MemorySpace, ExprPtr, uint64_t, uint64_t, Span>(), py::arg("memory_space"),
           py::arg("addr"), py::arg("size"), py::arg("id"), py::arg("span") = Span::unknown())
      .def_readwrite("memory_space_", &MemRef::memory_space_)
      .def_readwrite("addr_", &MemRef::addr_)
      .def_readwrite("size_", &MemRef::size_)
      .def_readwrite("id_", &MemRef::id_);

  // ConstInt
  auto constint_class = py::class_<ConstInt, Expr, std::shared_ptr<ConstInt>>(ir, "ConstInt");
  constint_class.def(py::init([](int64_t value, FwkDataType dtype, const Span &span) {
    return std::make_shared<ConstInt>(value, ConvertDataType(dtype), span);
  }), py::arg("value"), py::arg("dtype"), py::arg("span"));
  BindFields<ConstInt>(constint_class);
  constint_class.def_property_readonly("dtype", &ConstInt::dtype);

  // ConstFloat
  auto constfloat_class = py::class_<ConstFloat, Expr, std::shared_ptr<ConstFloat>>(ir, "ConstFloat");
  constfloat_class.def(py::init([](double value, FwkDataType dtype, const Span &span) {
    return std::make_shared<ConstFloat>(value, ConvertDataType(dtype), span);
  }), py::arg("value"), py::arg("dtype"), py::arg("span"));
  BindFields<ConstFloat>(constfloat_class);
  constfloat_class.def_property_readonly("dtype", &ConstFloat::dtype);

  // ConstBool
  auto constbool_class = py::class_<ConstBool, Expr, std::shared_ptr<ConstBool>>(ir, "ConstBool");
  constbool_class.def(py::init<bool, const Span &>(), py::arg("value"), py::arg("span"));
  BindFields<ConstBool>(constbool_class);
  constbool_class.def_property_readonly("dtype", &ConstBool::dtype);
  // Call
  auto call_class = py::class_<Call, Expr, std::shared_ptr<Call>>(ir, "Call");
  call_class.def(py::init<const OpPtr &, const std::vector<ExprPtr> &, const Span &>(),
                 py::arg("op"), py::arg("args"), py::arg("span"));
  call_class.def(py::init<const OpPtr &, const std::vector<ExprPtr> &, const TypePtr &, const Span &>(),
                 py::arg("op"), py::arg("args"), py::arg("type"), py::arg("span"));
  call_class.def(py::init([](const OpPtr &op, const std::vector<ExprPtr> &args, const py::dict &kwargs_dict,
                              const Span &span) {
    auto kwargs = ConvertKwargsDict(kwargs_dict);
    return std::make_shared<Call>(op, args, kwargs, span);
  }), py::arg("op"), py::arg("args"), py::arg("kwargs"), py::arg("span"));
  call_class.def(py::init([](const OpPtr &op, const std::vector<ExprPtr> &args, const py::dict &kwargs_dict,
                              const TypePtr &type, const Span &span) {
    auto kwargs = ConvertKwargsDict(kwargs_dict);
    return std::make_shared<Call>(op, args, kwargs, type, span);
  }), py::arg("op"), py::arg("args"), py::arg("kwargs"), py::arg("type"), py::arg("span"));
  BindFields<Call>(call_class);
  call_class.def_property_readonly("kwargs", [](const CallPtr &self) {
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
  auto make_tuple_class = py::class_<MakeTuple, Expr, std::shared_ptr<MakeTuple>>(ir, "MakeTuple");
  make_tuple_class.def(py::init<const std::vector<ExprPtr> &, const Span &>(),
                       py::arg("elements"), py::arg("span"));
  BindFields<MakeTuple>(make_tuple_class);

  // TupleGetItemExpr
  auto tuple_get_item_class = py::class_<TupleGetItemExpr, Expr, std::shared_ptr<TupleGetItemExpr>>(ir, "TupleGetItemExpr");
  tuple_get_item_class.def(py::init<const ExprPtr &, int, const Span &>(),
                           py::arg("tuple"), py::arg("index"), py::arg("span"));
  BindFields<TupleGetItemExpr>(tuple_get_item_class);

  // BinaryExpr
  auto binaryexpr_class = py::class_<BinaryExpr, Expr, std::shared_ptr<BinaryExpr>>(ir, "BinaryExpr");
  BindFields<BinaryExpr>(binaryexpr_class);

  // UnaryExpr
  auto unaryexpr_class = py::class_<UnaryExpr, Expr, std::shared_ptr<UnaryExpr>>(ir, "UnaryExpr");
  BindFields<UnaryExpr>(unaryexpr_class);

#define BIND_BINARY_EXPR(OpName, Description)                                                  \
  py::class_<OpName, BinaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description)            \
      .def(py::init([](const ExprPtr &left, const ExprPtr &right, FwkDataType dtype, const Span &span) { \
        return std::make_shared<OpName>(left, right, ConvertDataType(dtype), span); \
      }), py::arg("left"), py::arg("right"), py::arg("dtype"), py::arg("span"));

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
      .def(py::init([](const ExprPtr &operand, FwkDataType dtype, const Span &span) { \
        return std::make_shared<OpName>(operand, ConvertDataType(dtype), span); \
      }), py::arg("operand"), py::arg("dtype"), py::arg("span"));

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
  auto stmt_class = py::class_<Stmt, IRNode, std::shared_ptr<Stmt>>(ir, "Stmt");
  BindFields<Stmt>(stmt_class);

  auto assign_stmt_class = py::class_<AssignStmt, Stmt, std::shared_ptr<AssignStmt>>(ir, "AssignStmt");
  assign_stmt_class.def(py::init<const VarPtr &, const ExprPtr &, const Span &>(),
                        py::arg("var"), py::arg("value"), py::arg("span"));
  BindFields<AssignStmt>(assign_stmt_class);

  auto if_stmt_class = py::class_<IfStmt, Stmt, std::shared_ptr<IfStmt>>(ir, "IfStmt");
  if_stmt_class.def(py::init<const ExprPtr &, const StmtPtr &, const std::optional<StmtPtr> &,
                             const std::vector<VarPtr> &, const Span &>(),
                    py::arg("condition"), py::arg("then_body"), py::arg("else_body") = py::none(),
                    py::arg("return_vars"), py::arg("span"));
  BindFields<IfStmt>(if_stmt_class);

  auto yield_stmt_class = py::class_<YieldStmt, Stmt, std::shared_ptr<YieldStmt>>(ir, "YieldStmt");
  yield_stmt_class.def(py::init<const std::vector<ExprPtr> &, const Span &>(),
                       py::arg("value"), py::arg("span"));
  yield_stmt_class.def(py::init<const Span &>(), py::arg("span"));
  BindFields<YieldStmt>(yield_stmt_class);

  auto return_stmt_class = py::class_<ReturnStmt, Stmt, std::shared_ptr<ReturnStmt>>(ir, "ReturnStmt");
  return_stmt_class.def(py::init<const std::vector<ExprPtr> &, const Span &>(),
                        py::arg("value"), py::arg("span"));
  return_stmt_class.def(py::init<const Span &>(), py::arg("span"));
  BindFields<ReturnStmt>(return_stmt_class);
  auto for_stmt_class = py::class_<ForStmt, Stmt, std::shared_ptr<ForStmt>>(ir, "ForStmt");
  for_stmt_class.def(
      py::init<const VarPtr &, const ExprPtr &, const ExprPtr &, const ExprPtr &, const std::vector<IterArgPtr> &,
               const StmtPtr &, const std::vector<VarPtr> &, const Span &>(),
      py::arg("loop_var"), py::arg("start"), py::arg("stop"), py::arg("step"), py::arg("iter_args"),
      py::arg("body"), py::arg("return_vars"), py::arg("span"));
  BindFields<ForStmt>(for_stmt_class);

  auto seq_stmts_class = py::class_<SeqStmts, Stmt, std::shared_ptr<SeqStmts>>(ir, "SeqStmts");
  seq_stmts_class.def(py::init<const std::vector<StmtPtr> &, const Span &>(),
                      py::arg("stmts"), py::arg("span"));
  BindFields<SeqStmts>(seq_stmts_class);

  auto op_stmts_class = py::class_<OpStmts, Stmt, std::shared_ptr<OpStmts>>(ir, "OpStmts");
  op_stmts_class.def(py::init<const std::vector<StmtPtr> &, const Span &>(),
                     py::arg("stmts"), py::arg("span"));
  BindFields<OpStmts>(op_stmts_class);

  auto eval_stmt_class = py::class_<EvalStmt, Stmt, std::shared_ptr<EvalStmt>>(ir, "EvalStmt");
  eval_stmt_class.def(py::init<const ExprPtr &, const Span &>(), py::arg("expr"), py::arg("span"));
  BindFields<EvalStmt>(eval_stmt_class);

  // FunctionType enum
  py::enum_<FunctionType>(ir, "FunctionType")
      .value("Opaque", FunctionType::Opaque)
      .value("Orchestration", FunctionType::Orchestration)
      .value("InCore", FunctionType::InCore)
      .export_values();

  // Function
  auto function_class = py::class_<Function, IRNode, std::shared_ptr<Function>>(ir, "Function");
  function_class.def(py::init<const std::string &, const std::vector<VarPtr> &, const std::vector<TypePtr> &,
                              const StmtPtr &, const Span &, FunctionType>(),
                     py::arg("name"), py::arg("params"), py::arg("return_types"), py::arg("body"),
                     py::arg("span"), py::arg("type") = FunctionType::Opaque);
  BindFields<Function>(function_class);
  // Program
  auto program_class = py::class_<Program, IRNode, std::shared_ptr<Program>>(ir, "Program");
  program_class.def(py::init<const std::vector<FunctionPtr> &, const std::string &, const Span &>(),
                    py::arg("functions"), py::arg("name"), py::arg("span"));
  program_class.def("get_function", &Program::GetFunction, py::arg("name"));
  program_class.def("get_global_var", &Program::GetGlobalVar, py::arg("name"));
  program_class.def_property_readonly("functions",
      [](const std::shared_ptr<const Program> &self) {
        py::dict result;
        for (const auto &[gvar, func] : self->functions_) {
          result[py::cast(gvar)] = py::cast(func);
        }
        return result;
      });
  program_class.def_readonly("name", &Program::name_);
  program_class.def_readonly("span", &Program::span_);

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
