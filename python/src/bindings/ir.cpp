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
#include <pybind11/pytypes.h>
#include <cstdint>
#include <memory>

#include "ir/builder/ir_context.h"
#include "ir/builder/ir_builder.h"
#include "ir/function.h"
#include "ir/object.h"
#include "ir/program.h"
#include "ir/statement.h"
#include "ir/type.h"
#include "ir/utils.h"
#include "ir/value.h"

namespace py = pybind11;
using namespace pto;

namespace pypto {

static void IrBindEnum(py::module &m) {
    py::enum_<ObjectType>(m, "ObjectType")
        .value("Program", ObjectType::Program)
        .value("Function", ObjectType::Function)
        .value("Statement", ObjectType::Statement)
        .value("Operation", ObjectType::Operation)
        .value("Value", ObjectType::Value)
        .value("Memory", ObjectType::Memory);

    py::enum_<DataType>(m, "DataType")
        .value("bool", DataType::BOOL)
        .value("int8", DataType::INT8)
        .value("int16", DataType::INT16)
        .value("int32", DataType::INT32)
        .value("int64", DataType::INT64)
        .value("uint8", DataType::UINT8)
        .value("uint16", DataType::UINT16)
        .value("uint32", DataType::UINT32)
        .value("uint64", DataType::UINT64)
        .value("float16", DataType::FP16)
        .value("bfloat16", DataType::BF16)
        .value("float", DataType::FP32)
        .value("double", DataType::FP64)
        .value("float8_e4m3fn", DataType::FP8_E4M3FN)
        .value("float8_e5m2", DataType::FP8_E5M2)
        .export_values()
        .def("bits", [](DataType dtype) { return DTypeInfoOf(dtype).bits; })
        .def("bytes", [](DataType dtype) { return DTypeInfoOf(dtype).bytes; })
        .def("is_float", [](DataType dtype) { return DTypeInfoOf(dtype).isFloat; })
        .def("__str__", [](DataType dtype) { return DTypeInfoOf(dtype).name; });

    py::enum_<FunctionKind>(m, "FunctionKind")
        .value("ControlFlow", FunctionKind::ControlFlow)
        .value("DataFlow", FunctionKind::DataFlow)
        .value("Kernel", FunctionKind::Kernel);

    py::enum_<StatementKind>(m, "StatementKind")
        .value("Compound", StatementKind::Compound)
        .value("Op", StatementKind::Op)
        .value("For", StatementKind::For)
        .value("If", StatementKind::If)
        .value("Yield", StatementKind::Yield)
        .value("Call", StatementKind::Call)
        .value("Return", StatementKind::Return);

    py::enum_<ValueKind>(m, "ValueKind")
        .value("Scalar", ValueKind::Scalar)
        .value("Tensor", ValueKind::Tensor)
        .value("T", ValueKind::Tile);

    py::enum_<MemSpaceKind>(m, "MemSpaceKind")
        .value("DDR", MemSpaceKind::DDR)
        .value("L2", MemSpaceKind::L2)
        .value("UB", MemSpaceKind::UB)
        .value("L1", MemSpaceKind::L1)
        .value("L0A", MemSpaceKind::L0A)
        .value("L0B", MemSpaceKind::L0B)
        .value("L0C", MemSpaceKind::L0C)
        .value("REG", MemSpaceKind::REG)
        .value("SHMEM", MemSpaceKind::SHMEM);

    py::enum_<Format>(m, "Format")
        .value("ND", Format::ND)
        .value("NZ", Format::NZ);
}

static void IrBindType(py::module &m) {
    py::class_<Type>(m, "Type")
        .def_property_readonly("dtype", &Type::GetDataType)
        .def("elem_size", [](const Type &self) { return self.GetDataTypeSize(); })
        .def("size", &Type::GetTypeSize);

    py::class_<ScalarType, Type>(m, "ScalarType")
        .def(py::init<DataType>(), py::arg("dtype"))
        .def("size", &ScalarType::GetTypeSize);

    py::class_<pto::TileType, Type>(m, "TileType")
        .def(py::init<DataType, const std::vector<int64_t> &>(), py::arg("dtype"), py::arg("shape"))
        .def_property_readonly("shape", &TileType::GetShape)
        .def("size", &TileType::GetTypeSize);

    py::class_<pto::TensorType, Type>(m, "TensorType")
        .def(py::init<DataType>(), py::arg("dtype"))
        .def("size", &TensorType::GetTypeSize);
}

static void IrBindObjClass(py::module &m) {
    py::class_<Object>(m, "Object")
        .def_property_readonly("id", &Object::GetID)
        .def_property_readonly("name", &Object::GetName)
        .def_property_readonly("type", &Object::GetObjectType)
        .def("properties", py::overload_cast<>(&Object::Attributes, py::const_));
}

static void IrBindValue(py::module &m) {
    py::class_<Value>(m, "Value")
        .def("ssaname", &Value::GetSSAName)
        .def_property_readonly("kind", &Value::GetValueKind)
        .def_property_readonly("type", &Value::GetType);

    py::class_<ScalarValue, Value>(m, "Scalar")
        .def(py::init([](DataType dtype, py::object val, std::string name = "") {
            if (val.is_none()) {
                return std::make_unique<ScalarValue>(dtype, name);
            } else if (py::isinstance<py::int_>(val)) {
                auto intVal = py::cast<int64_t>(val);
                if (DTypeInfoOf(dtype).isFloat) {
                    return std::make_unique<ScalarValue>(dtype, name, ScalarValueKind::Immediate, static_cast<double>(intVal));
                } else {
                    return std::make_unique<ScalarValue>(dtype, name, ScalarValueKind::Immediate, intVal);
                }
            } else if (py::isinstance<py::float_>(val)) {
                auto fval = py::cast<double>(val);
                if (DTypeInfoOf(dtype).isFloat) {
                    return std::make_unique<ScalarValue>(dtype, name, ScalarValueKind::Immediate, fval);
                } else {
                    return std::make_unique<ScalarValue>(dtype, name, ScalarValueKind::Immediate, static_cast<int64_t>(fval));
                }
            } else {
                throw py::type_error("Unsupported value type for key: " + name);
            }
        }), py::arg("type"), py::arg("val"), py::arg("name") = "")
        .def_property_readonly("type", &ScalarValue::GetType)
        .def("is_constant", &ScalarValue::HasImmediateValue)
        .def("value", [](const ScalarValue &self) {
            if (self.HasImmediateValue()) {
                const auto &val = self.GetImmediateValue();
                if (self.GetDataType() == DataType::BOOL) {
                    return py::cast(static_cast<bool>(std::get<int64_t>(val)));
                } else if (DTypeInfoOf(self.GetDataType()).isFloat) {
                    return py::cast(std::get<double>(val));
                } else {
                    return py::cast(std::get<int64_t>(val));
                }
            } else {
                return py::cast(self.GetSSAName());
            }
        });

    py::class_<Memory, Object>(m, "Memory")
        .def(py::init<uint64_t, MemSpaceKind>(), py::arg("size"), py::arg("kind"))
        .def_property("size", &Memory::GetSize, &Memory::SetSize)
        .def_property("kind", &Memory::GetSpace, &Memory::SetSpace)
        .def_property("addr", &Memory::GetAddr, &Memory::SetAddr);

    py::class_<TileValue, Value>(m, "Tile")
        .def(py::init<std::string, std::vector<ScalarValuePtr>, std::vector<int64_t>, std::vector<int64_t>,
                 ScalarValuePtr, DataType, MemoryPtr>(),
            py::arg("name"), py::arg("valid_shape"), py::arg("shape"), py::arg("strides"), py::arg("offset"),
            py::arg("dtype"), py::arg("memory"))
        .def_property("shape", &TileValue::GetShape, &TileValue::SetShape)
        .def_property("strides", &TileValue::GetStrides, &TileValue::SetStrides)
        .def_property("offset", &TileValue::GetStartOffset, &TileValue::SetStartOffset)
        .def_property("memory", &TileValue::GetMemory, &TileValue::SetMemory);

    py::class_<TensorValue, Value>(m, "Tensor")
        .def(py::init<std::vector<ScalarValuePtr>, DataType, std::string, Format>(), py::arg("shape"), py::arg("dtype"),
            py::arg("name"), py::arg("format"))
        .def_property_readonly("shape", &TensorValue::GetShape)
        .def_property("format", &TensorValue::GetFormat, &TensorValue::SetFormat);
}

static void IrBindStatement(py::module &m) {
    py::class_<Statement, Object>(m, "Statement")
        .def_property_readonly("kind", &Statement::GetKind);

    py::class_<CompoundStatement, Statement>(m, "CompoundStatement")
        .def(py::init<>())
        .def(py::init<CompoundStatementPtr>(), py::arg("parent"))
        .def_property("parent", &CompoundStatement::GetParent, &CompoundStatement::SetParent)
        .def("stmts", &CompoundStatement::GetStatements, py::return_value_policy::reference_internal)
        .def("get_var", &CompoundStatement::GetEnvVar, py::arg("name"))
        .def("set_var", &CompoundStatement::SetEnvVar, py::arg("name"), py::arg("value"));

    py::class_<OpStatement, Statement>(m, "OpStatement")
        .def(py::init<>())
        .def(
            "operations", [](OpStatement &self) { return self.Operations(); },
            py::return_value_policy::reference_internal);

    py::class_<YieldStatement, Statement>(m, "YieldStatement")
        .def(py::init<>())
        .def(
            "values", [](const YieldStatement &self) { return self.Values(); },
            py::return_value_policy::reference_internal);

    py::class_<ForStatement, Statement>(m, "ForStatement")
        .def(py::init<ScalarValuePtr, ScalarValuePtr, ScalarValuePtr, ScalarValuePtr>(), py::arg("var"),
            py::arg("start"), py::arg("end"), py::arg("step"))
        .def_property_readonly("var", &ForStatement::GetIterationVar)
        .def_property_readonly("start", &ForStatement::GetStart)
        .def_property_readonly("end", &ForStatement::GetEnd)
        .def_property_readonly("step", &ForStatement::GetStep)
        .def(
            "stmts", [](ForStatement &self) { return self.GetCompound(); }, py::return_value_policy::reference_internal)
        .def("yield", [](ForStatement &self) { return self.Yield(); });

    py::class_<IfStatement, Statement>(m, "IfStatement")
        .def(py::init<ScalarValuePtr>(), py::arg("cond"))
        .def("condition", &IfStatement::GetCondition)
        .def("then_stmts",
            [](IfStatement &self) { return self.GetThenCompound(); },
            py::return_value_policy::reference_internal)
        .def("else_stmts",
            [](IfStatement &self) { return self.GetElseCompound(); },
            py::return_value_policy::reference_internal)
        .def("results", [](const IfStatement &self) { return self.Results(); });

    py::class_<ReturnStatement, Statement>(m, "ReturnStatement")
        .def(py::init<>())
        .def("values",
            [](ReturnStatement &self) { return self.Values(); },
            py::return_value_policy::reference_internal);
}

static void IrBindModule(py::module &m) {
    py::class_<ProgramModule, Object>(m, "module")
        .def(py::init<const std::string &>(), py::arg("name"))
        .def_property("entry", &ProgramModule::GetProgramEntry, &ProgramModule::SetProgramEntry)
        .def("functions", &ProgramModule::GetFunctions, py::return_value_policy::reference_internal);
}

static void IrBindFunction(py::module &m) {
    py::class_<FunctionSignature>(m, "FunctionSignature")
        .def(py::init<>())
        .def_readwrite("arguments", &FunctionSignature::arguments, py::return_value_policy::reference_internal)
        .def_readwrite("returns", &FunctionSignature::results, py::return_value_policy::reference_internal);

    py::class_<Function, Object>(m, "Function")
        .def(py::init([](FunctionKind kind, const std::string &name, FunctionSignature sig) {
            return std::make_unique<Function>(name, kind, sig);
        }),
            py::arg("kind"), py::arg("name"), py::arg("sig"))
        .def(
            "stmts", [](Function &self) { return self.GetCompound(); }, py::return_value_policy::reference_internal)
        .def_property_readonly("kind", &Function::GetKind);
}

static void IrBuilderBindOp(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_unary_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr in, ScalarValuePtr out) {
                return self.CreateUnaryScalarOp(opcode, in, out);
            },
            py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def(
            "create_binary_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr lhs, ScalarValuePtr rhs, ScalarValuePtr out) {
                return self.CreateBinaryScalarOp(opcode, lhs, rhs, out);
            },
            py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"));

    irBuilder
        .def(
            "create_unary_op",
            [](IRBuilder &self, Opcode opcode, ValuePtr in, ValuePtr out) {
                return self.CreateUnaryOp(opcode, in, out);
            },
            py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def(
            "create_binary_op",
            [](IRBuilder &self, Opcode opcode, ValuePtr lhs, ValuePtr rhs, ValuePtr out) {
                return self.CreateBinaryOp(opcode, lhs, rhs, out);
            },
            py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"));
}

static void IrBindBuilder(py::module &m) {
    py::class_<IRBuilderContext>(m, "IrBuilderContext")
        .def("push_scope", &IRBuilderContext::PushScope)
        .def("pop_scope", &IRBuilderContext::PopScope);

    auto irBuilder = py::class_<IRBuilder>(m, "IrBuilder")
        .def(py::init<std::shared_ptr<ProgramModule>>(), py::arg("module") = nullptr)
        .def("create_function", &IRBuilder::CreateFunction, py::arg("name"), py::arg("kind"), py::arg("sig"),
            py::arg("is_entry") = false)
        .def("create_tensor", &IRBuilder::CreateTensor, py::arg("ctx"), py::arg("shape"), py::arg("dtype"),
            py::arg("name") = "")
        .def("create_tile", &IRBuilder::CreateTile, py::arg("ctx"), py::arg("shape"), py::arg("dtype"),
            py::arg("name") = "")
        .def("create_scalar", &IRBuilder::CreateScalar, py::arg("ctx"), py::arg("dtype"), py::arg("name") = "")
        .def("create_const",
            [](IRBuilder &self, IRBuilderContext &ctx, py::object value, const std::string &name) {
                if (py::isinstance<int64_t>(value)) {
                    return self.CreateConst(ctx, py::cast<int64_t>(value), name);
                } else if (py::isinstance<double>(value)) {
                    return self.CreateConst(ctx, py::cast<double>(value), name);
                } else {
                    throw py::type_error("Unsupported const value type");
                }
            },
            py::arg("ctx"), py::arg("value"), py::arg("name") = "")
        .def("create_op", &IRBuilder::CreateOpStmt, py::arg("ctx"))
        .def("create_for", &IRBuilder::CreateForStmt, py::arg("ctx"), py::arg("var"), py::arg("start"), py::arg("end"),
            py::arg("step"))
        .def("create_if", &IRBuilder::CreateIfStmt, py::arg("ctx"), py::arg("cond"))
        .def("create_return", &IRBuilder::CreateReturn, py::arg("ctx"), py::arg("values"))
        .def("enter_function",
            [](IRBuilder &self, IRBuilderContext &ctx, std::shared_ptr<Function> func) {
                self.EnterFunctionBody(ctx, func);
            },
            py::arg("ctx"), py::arg("func"))
        .def("enter_for",
            [](IRBuilder &self, IRBuilderContext &ctx, ForStatementPtr stmt) { self.EnterForBody(ctx, stmt); },
            py::arg("ctx"), py::arg("stmt"))
        .def("enter_if_then",
            [](IRBuilder &self, IRBuilderContext &ctx, IfStatementPtr stmt) { self.EnterIfThen(ctx, stmt); },
            py::arg("ctx"), py::arg("stmt"))
        .def("enter_if_else",
            [](IRBuilder &self, IRBuilderContext &ctx, IfStatementPtr stmt) { self.EnterIfElse(ctx, stmt); },
            py::arg("ctx"), py::arg("stmt"))
        .def("exit_for",
            [](IRBuilder &self, IRBuilderContext &ctx, ForStatementPtr stmt) { self.ExitForStatement(ctx, stmt); },
            py::arg("ctx"), py::arg("stmt"))
        .def("exit_if",
            [](IRBuilder &self, IRBuilderContext &ctx, IfStatementPtr stmt) { self.ExitIfStatement(ctx, stmt); },
            py::arg("ctx"), py::arg("stmt"))
        .def("emit", &IRBuilder::Emit, py::arg("ctx"), py::arg("operation"));

    IrBuilderBindOp(irBuilder);
}

void BindIr(py::module &m) {
    auto ir = m.def_submodule("ir", "IR module");
    IrBindEnum(ir);
    IrBindObjClass(ir);
    IrBindType(ir);
    IrBindValue(ir);
    IrBindStatement(ir);
    IrBindFunction(ir);
    IrBindModule(ir);
    IrBindBuilder(ir);
}
} // namespace pypto
