/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <nanobind/nanobind.h>
#include "nb_common.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "bindings.h"
#include "ir/builder/ir_context.h"
#include "ir/builder/ir_builder.h"
#include "ir/function.h"
#include "ir/object.h"
#include "ir/operation_base.h"
#include "ir/program.h"
#include "ir/statement.h"
#include "ir/type.h"
#include "ir/utils.h"
#include "ir/utils_defop.h"
#include "ir/value.h"
#include "ir/block_call.h"
namespace nb = nanobind;

namespace pto {

static void IrBindEnum(nb::module_ &m) {
    nb::enum_<ObjectType>(m, "ObjectType", nb::is_arithmetic())
        .value("Program", ObjectType::Program)
        .value("Function", ObjectType::Function)
        .value("Statement", ObjectType::Statement)
        .value("Operation", ObjectType::Operation)
        .value("Value", ObjectType::Value)
        .value("Memory", ObjectType::Memory);

    nb::enum_<DataType>(m, "DataType", nb::is_arithmetic())
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
        .value("float32", DataType::FP32)
        .value("float", DataType::FP32)
        .value("float64", DataType::FP64)
        .value("double", DataType::FP64)
        .value("float8_e4m3fn", DataType::FP8_E4M3FN)
        .value("float8_e5m2", DataType::FP8_E5M2)
        .export_values()
        .def("bits", [](DataType dtype) { return DTypeInfoOf(dtype).bits; })
        .def("bytes", [](DataType dtype) { return DTypeInfoOf(dtype).bytes; })
        .def("is_float", [](DataType dtype) { return DTypeInfoOf(dtype).isFloat; })
        .def("__str__", [](DataType dtype) { return DTypeInfoOf(dtype).name; });

    nb::enum_<FunctionKind>(m, "FunctionKind", nb::is_arithmetic())
        .value("ControlFlow", FunctionKind::ControlFlow)
        .value("DataFlow", FunctionKind::DataFlow)
        .value("Block", FunctionKind::Block);

    nb::enum_<StatementKind>(m, "StatementKind", nb::is_arithmetic())
        .value("Compound", StatementKind::Compound)
        .value("Op", StatementKind::Op)
        .value("For", StatementKind::For)
        .value("If", StatementKind::If)
        .value("Yield", StatementKind::Yield)
        .value("Call", StatementKind::Call)
        .value("Return", StatementKind::Return);

    nb::enum_<ValueKind>(m, "ValueKind", nb::is_arithmetic())
        .value("Scalar", ValueKind::Scalar)
        .value("Tensor", ValueKind::Tensor)
        .value("Tile", ValueKind::Tile);

    nb::enum_<MemSpaceKind>(m, "MemSpaceKind", nb::is_arithmetic())
        .value("DDR", MemSpaceKind::DDR)
        .value("L2", MemSpaceKind::L2)
        .value("UB", MemSpaceKind::UB)
        .value("L1", MemSpaceKind::L1)
        .value("L0A", MemSpaceKind::L0A)
        .value("L0B", MemSpaceKind::L0B)
        .value("L0C", MemSpaceKind::L0C)
        .value("REG", MemSpaceKind::REG)
        .value("SHMEM", MemSpaceKind::SHMEM);

    nb::enum_<Format>(m, "Format", nb::is_arithmetic()).value("ND", Format::ND).value("NZ", Format::NZ);

    nb::enum_<Opcode>(m, "Opcode", nb::is_arithmetic())
#define DEFOP DEFOP_OPCODE_PYENUM
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP
        ;
}

static void IrBindType(nb::module_ &m) {
    nb::class_<Type>(m, "Type")
        .def_prop_ro("dtype", &Type::GetDataType)
        .def("elem_size", [](const Type &self) { return self.GetDataTypeSize(); })
        .def("size", &Type::GetTypeSize);

    nb::class_<ScalarType, Type>(m, "ScalarType")
        .def(nb::init<DataType>(), nb::arg("dtype"))
        .def("size", &ScalarType::GetTypeSize);

    nb::class_<pto::TileType, Type>(m, "TileType")
        .def(nb::init<DataType, const std::vector<int64_t> &>(), nb::arg("dtype"), nb::arg("shape"))
        .def_prop_ro("shape", &TileType::GetShape)
        .def("size", &TileType::GetTypeSize);

    nb::class_<pto::TensorType, Type>(m, "TensorType")
        .def(nb::init<DataType>(), nb::arg("dtype"))
        .def("size", &TensorType::GetTypeSize);
}

static void IrBindObjClass(nb::module_ &m) {
    nb::class_<Object>(m, "Object")
        .def_prop_ro("id", &Object::GetID)
        .def_prop_ro("name", &Object::GetName)
        .def_prop_ro("type", &Object::GetObjectType)
        .def("properties", nb::overload_cast<>(&Object::Attributes, nb::const_));
}

struct Pet {
    Pet(const std::string &name, nb::object age) : name_(name) { (void)age; }

    void set(int age) { age_ = age; }
    void set(const std::string &name) { name_ = name; }

    std::string name_;
    int age_;
};

static void IrBindValue(nb::module_ &m) {
    nb::class_<Value, Object>(m, "Value")
        .def("ssaname", &Value::GetSSAName)
        .def_prop_ro("kind", &Value::GetValueKind)
        .def_prop_ro("type", &Value::GetType);

    nb::class_<ScalarValue, Value>(m, "Scalar")
        .def("__init__",
            [](ScalarValue *self, DataType dtype, nb::object &val, const std::string &name) {
                if (val.is_none()) {
                    new (self)ScalarValue(dtype, name);
                } else if (nb::isinstance<nb::int_>(val)) {
                    auto intVal = nb::cast<int64_t>(val);
                    if (DTypeInfoOf(dtype).isFloat) {
                        new (self)ScalarValue(dtype, name, ScalarValueKind::Immediate, static_cast<double>(intVal));
                    } else {
                        new (self)ScalarValue(dtype, name, ScalarValueKind::Immediate, intVal);
                    }
                } else if (nb::isinstance<nb::float_>(val)) {
                    auto fval = nb::cast<double>(val);
                    if (DTypeInfoOf(dtype).isFloat) {
                        new (self)ScalarValue(dtype, name, ScalarValueKind::Immediate, fval);
                    } else {
                        new (self)ScalarValue(dtype, name, ScalarValueKind::Immediate, static_cast<int64_t>(fval));
                    }
                } else {
                    throw nb::type_error(("Unsupported value type for key: " + name).c_str());
                }
            }, nb::arg("dtype"), nb::arg("val").none(), nb::arg("name") = "")
        .def_prop_ro("type", &ScalarValue::GetType)
        .def("is_constant", &ScalarValue::HasImmediateValue)
        .def("value", [](const ScalarValue &self) {
            if (self.HasImmediateValue()) {
                const auto &val = self.GetImmediateValue();
                if (self.GetDataType() == DataType::BOOL) {
                    return nb::cast(static_cast<bool>(std::get<int64_t>(val)));
                } else if (DTypeInfoOf(self.GetDataType()).isFloat) {
                    return nb::cast(std::get<double>(val));
                } else {
                    return nb::cast(std::get<int64_t>(val));
                }
            } else {
                return nb::cast(self.GetSSAName());
            }
        });

    nb::class_<Memory, Object>(m, "Memory")
        .def(nb::init<uint64_t, MemSpaceKind>(), nb::arg("size"), nb::arg("kind"))
        .def_prop_rw("size", &Memory::GetSize, &Memory::SetSize)
        .def_prop_rw("kind", &Memory::GetSpace, &Memory::SetSpace)
        .def_prop_rw("addr", &Memory::GetAddr, &Memory::SetAddr);

    nb::class_<TileValue, Value>(m, "Tile")
        .def(nb::init<std::string, std::vector<ScalarValuePtr>, std::vector<int64_t>, std::vector<int64_t>,
                 ScalarValuePtr, DataType, MemoryPtr>(),
            nb::arg("name"), nb::arg("valid_shape"), nb::arg("shape"), nb::arg("strides"), nb::arg("offset"),
            nb::arg("dtype"), nb::arg("memory"))
        .def("set_valid_shape", &TileValue::SetValidShape, nb::arg("new_valid_shape"))
        .def("set_memory_param", &TileValue::SetMemoryParam, nb::arg("byte_size"), nb::arg("space"), nb::arg("addr"))
        .def("set_memory_addr", &TileValue::SetMemoryAddr, nb::arg("addr"))
        .def_prop_rw("shape", &TileValue::GetShape, &TileValue::SetShape)
        .def_prop_rw("strides", &TileValue::GetStrides, &TileValue::SetStrides)
        .def_prop_rw("offset", &TileValue::GetStartOffset, &TileValue::SetStartOffset)
        .def_prop_rw("memory", &TileValue::GetMemory, &TileValue::SetMemory);

    nb::class_<TensorValue, Value>(m, "Tensor")
        .def(nb::init<std::vector<ScalarValuePtr>, DataType, std::string, Format>(), nb::arg("shape"), nb::arg("dtype"),
            nb::arg("name"), nb::arg("format"))
        .def_prop_ro("shape", &TensorValue::GetShape)
        .def_prop_rw("format", &TensorValue::GetFormat, &TensorValue::SetFormat);
}

static void IrBindOperation(nb::module_ &m) {
    nb::class_<Operation, Object>(m, "Operation")
        .def_prop_ro("opcode", &Operation::GetOpcode)
        .def("ioperands", &Operation::GetIOperands, nb::rv_policy::reference_internal)
        .def("ooperands", &Operation::GetOOperands, nb::rv_policy::reference_internal);

    nb::class_<UnaryOp, Operation>(m, "UnaryOp")
        .def_prop_rw("combine_axis", &UnaryOp::GetCombineAxis, &UnaryOp::SetCombineAxis);
    nb::class_<BinaryOp, Operation>(m, "BinaryOp")
        .def_prop_rw("combine_axis", &BinaryOp::GetCombineAxis, &BinaryOp::SetCombineAxis);
    nb::class_<BinaryScalarMixOp, Operation>(m, "BinaryScalarMixOp")
        .def_prop_rw("combine_axis", &BinaryScalarMixOp::GetCombineAxis, &BinaryScalarMixOp::SetCombineAxis)
        .def_prop_rw("reverse", &BinaryScalarMixOp::GetReverse, &BinaryScalarMixOp::SetReverse);

    nb::class_<UnaryScalarOp, Operation>(m, "UnaryScalarOp");
    nb::class_<BinaryScalarOp, Operation>(m, "BinaryScalarOp");
}

static void IrBindStatement(nb::module_ &m) {
    nb::class_<Statement, Object>(m, "Statement").def_prop_ro("kind", &Statement::GetKind);

    nb::class_<CompoundStatement, Statement>(m, "CompoundStatement")
        .def(nb::init<>())
        .def(nb::init<CompoundStatementPtr>(), nb::arg("parent"))
        .def_prop_rw("parent", &CompoundStatement::GetParent, &CompoundStatement::SetParent)
        .def(
            "stmts", [](CompoundStatement &self) { return self.GetStatements(); }, nb::rv_policy::reference_internal)
        .def("vars", [](CompoundStatement &self) { return self.GetEnvTable(); }, nb::rv_policy::reference_internal);

    nb::class_<OpStatement, Statement>(m, "OpStatement")
        .def(nb::init<>())
        .def("operations", [](OpStatement &self) { return self.Operations(); }, nb::rv_policy::reference_internal);

    nb::class_<YieldStatement, Statement>(m, "YieldStatement")
        .def(nb::init<>())
        .def("values", [](const YieldStatement &self) { return self.Values(); }, nb::rv_policy::reference_internal);

    nb::class_<ForStatement, Statement>(m, "ForStatement")
        .def(nb::init<ScalarValuePtr, ScalarValuePtr, ScalarValuePtr, ScalarValuePtr>(), nb::arg("var"),
            nb::arg("start"), nb::arg("end"), nb::arg("step"))
        .def_prop_ro("var", &ForStatement::GetIterationVar)
        .def_prop_ro("start", &ForStatement::GetStart)
        .def_prop_ro("end", &ForStatement::GetEnd)
        .def_prop_ro("step", &ForStatement::GetStep)
        .def(
            "stmts", [](ForStatement &self) { return self.GetCompound(); }, nb::rv_policy::reference_internal)
        .def("yield", [](ForStatement &self) { return self.Yield(); });

    nb::class_<IfStatement, Statement>(m, "IfStatement")
        .def(nb::init<ScalarValuePtr>(), nb::arg("cond"))
        .def("condition", &IfStatement::GetCondition)
        .def(
            "then_stmts", [](IfStatement &self) { return self.GetThenCompound(); }, nb::rv_policy::reference_internal)
        .def(
            "else_stmts", [](IfStatement &self) { return self.GetElseCompound(); }, nb::rv_policy::reference_internal)
        .def("results", [](const IfStatement &self) { return self.Results(); });

    nb::class_<ReturnStatement, Statement>(m, "ReturnStatement")
        .def(nb::init<>())
        .def("values", [](ReturnStatement &self) { return self.Values(); }, nb::rv_policy::reference_internal);
}

static void IrBindModule(nb::module_ &m) {
    nb::class_<ProgramModule, Object>(m, "module")
        .def(nb::init<const std::string &>(), nb::arg("name"))
        .def_prop_rw("entry", &ProgramModule::GetProgramEntry, &ProgramModule::SetProgramEntry)
        .def("functions", &ProgramModule::GetFunctions, nb::rv_policy::reference_internal)
        .def("add_function", &ProgramModule::AddFunction, nb::arg("function"));
}

static void IrBindFunction(nb::module_ &m) {
    nb::class_<FunctionSignature>(m, "FunctionSignature")
        .def(nb::init<>())
        .def(nb::init<const std::vector<pto::TensorValuePtr> &>(), nb::arg("args"))
        .def(nb::init<const std::vector<pto::TensorValuePtr> &, const std::vector<pto::TensorValuePtr> &>(),
            nb::arg("input_args"), nb::arg("output_args"))
        .def_rw("arguments", &FunctionSignature::arguments, nb::rv_policy::reference_internal)
        .def_rw("returns", &FunctionSignature::results, nb::rv_policy::reference_internal);

    nb::class_<Function, Object>(m, "Function")
        .def_static(
            "create",
            [](FunctionKind kind, const std::string &name, FunctionSignature sig) {
                return std::make_shared<Function>(name, kind, sig);
            },
            nb::arg("kind"), nb::arg("name"), nb::arg("sig"))
        .def("get_sig", &Function::GetSignature)
        .def(
            "stmts", [](Function &self) { return self.GetCompound(); }, nb::rv_policy::reference_internal)
        .def_prop_ro("kind", &Function::GetKind);
}

// Scalar operations (from operation.def)
static void IrBuilderBindScalarOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_unary_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr in, ScalarValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateUnaryScalarOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_binary_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr lhs, ScalarValuePtr rhs, ScalarValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateBinaryScalarOp(opcode, lhs, rhs, out));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"))
        .def(
            "create_call_1_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr arg0, ScalarValuePtr out, const std::string &name) {
                return std::static_pointer_cast<Operation>(self.CreateCall1ScalarOp(opcode, arg0, out, name));
            },
            nb::arg("opcode"), nb::arg("ar0"), nb::arg("out"), nb::arg("name"))
        .def(
            "create_call_2_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr arg0, ScalarValuePtr arg1, ScalarValuePtr out,
                const std::string &name) {
                return std::static_pointer_cast<Operation>(self.CreateCall2ScalarOp(opcode, arg0, arg1, out, name));
            },
            nb::arg("opcode"), nb::arg("arg0"), nb::arg("arg1"), nb::arg("out"), nb::arg("name"))
        .def(
            "create_call_3_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr arg0, ScalarValuePtr arg1, ScalarValuePtr arg2,
                ScalarValuePtr out, const std::string &name) {
                return std::static_pointer_cast<Operation>(
                    self.CreateCall3ScalarOp(opcode, arg0, arg1, arg2, out, name));
            },
            nb::arg("opcode"), nb::arg("arg0"), nb::arg("arg1"), nb::arg("arg2"), nb::arg("out"), nb::arg("name"))
        .def(
            "create_call_4_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr arg0, ScalarValuePtr arg1, ScalarValuePtr arg2,
                ScalarValuePtr arg3, ScalarValuePtr out, const std::string &name) {
                return std::static_pointer_cast<Operation>(
                    self.CreateCall4ScalarOp(opcode, arg0, arg1, arg2, arg3, out, name));
            },
            nb::arg("opcode"), nb::arg("arg0"), nb::arg("arg1"), nb::arg("arg2"), nb::arg("arg3"), nb::arg("out"),
            nb::arg("name"))
        .def(
            "create_call_5_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr arg0, ScalarValuePtr arg1, ScalarValuePtr arg2,
                ScalarValuePtr arg3, ScalarValuePtr arg4, ScalarValuePtr out, const std::string &name) {
                return std::static_pointer_cast<Operation>(
                    self.CreateCall5ScalarOp(opcode, arg0, arg1, arg2, arg3, arg4, out, name));
            },
            nb::arg("opcode"), nb::arg("arg0"), nb::arg("arg1"), nb::arg("arg2"), nb::arg("arg3"), nb::arg("arg4"),
            nb::arg("out"), nb::arg("name"));
}

// Tile operations (from tile_graph.def)
static void IrBuilderBindUnaryOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_unary_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateUnaryOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_unary_with_temp_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateUnaryWithTempOp(opcode, in, out, temp));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"), nb::arg("temp"))
        .def(
            "create_expand_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateExpandOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_transpose_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateTransposeOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_transpose_vnchw_conv_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateTransposeVnchwConvOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_onehot_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateOnehotOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_cast_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateCastOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_cumsum_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateCumSumOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_convert_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateConvertOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_duplicate_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateDuplicateOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_any_data_copy_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateAnyDataCopyOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_argsort_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateArgSortOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"));
}

static void IrBuilderBindBinaryOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_binary_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateBinaryOp(opcode, lhs, rhs, out));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"))
        .def(
            "create_binary_with_temp_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out,
                TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateBinaryWithTempOp(opcode, lhs, rhs, out, temp));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"), nb::arg("temp"))
        .def(
            "create_binary_scalar_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, ScalarValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateBinaryScalarMixOp(opcode, lhs, rhs, out));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("scalar"), nb::arg("out"))
        .def(
            "create_scatter_elements_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr src0, TileValuePtr src1, ScalarValuePtr scatter,
                TileValuePtr out) {
                return std::static_pointer_cast<Operation>(
                    self.CreateScatterElementsOp(opcode, src0, src1, scatter, out));
            },
            nb::arg("opcode"), nb::arg("src0"), nb::arg("src1"), nb::arg("scatter"), nb::arg("out"))
        .def(
            "create_scatter_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr src0, TileValuePtr src1, TileValuePtr src2,
                TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateScatterOp(opcode, src0, src1, src2, out));
            },
            nb::arg("opcode"), nb::arg("src0"), nb::arg("src1"), nb::arg("src2"), nb::arg("out"))
        .def(
            "create_gather_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateGatherOp(opcode, lhs, rhs, out));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"))
        .def(
            "create_gather_extended_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateGatherExtendedOp(opcode, lhs, rhs, out));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"))
        .def(
            "create_broadcast_with_temp_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out,
                TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateBroadcastWithTempOp(opcode, lhs, rhs, out, temp));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"), nb::arg("temp"))
        .def(
            "create_fused_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateFusedOp(opcode, lhs, rhs, out));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"))
        .def(
            "create_load_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateLoadOp(opcode, lhs, rhs, out));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"))
        .def(
            "create_copy_in_out_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateCopyInOutOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"));
}

static void IrBuilderBindScalarInputOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_range_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr start, ScalarValuePtr step, ScalarValuePtr size,
                TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateRangeOp(opcode, start, step, size, out));
            },
            nb::arg("opcode"), nb::arg("start"), nb::arg("step"), nb::arg("size"), nb::arg("out"))
        .def(
            "create_vec_dup_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr value, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateVecDupOp(opcode, value, out));
            },
            nb::arg("opcode"), nb::arg("value"), nb::arg("out"))
        .def(
            "create_pow_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, ScalarValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreatePowOp(opcode, lhs, rhs, out));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"));
}

static void IrBuilderBindScalarListOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_matmul_load_op",
            [](IRBuilder &self, Opcode opcode, TensorValuePtr input, const std::vector<ScalarValuePtr> &offsets,
                TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulLoadOp(opcode, input, offsets, out));
            },
            nb::arg("opcode"), nb::arg("input"), nb::arg("offsets"), nb::arg("out"))
        .def(
            "create_matmul_extract_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input, const std::vector<ScalarValuePtr> &offsets,
                TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulExtractOp(opcode, input, offsets, out));
            },
            nb::arg("opcode"), nb::arg("input"), nb::arg("offsets"), nb::arg("out"))
        .def(
            "create_matmul_store_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input, const std::vector<ScalarValuePtr> &offsets,
                TensorValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulStoreOp(opcode, input, offsets, out));
            },
            nb::arg("opcode"), nb::arg("input"), nb::arg("offsets"), nb::arg("out"))
        .def(
            "create_matmul_bias_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input, const std::vector<ScalarValuePtr> &offsets,
                TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulBiasOp(opcode, input, offsets, out));
            },
            nb::arg("opcode"), nb::arg("input"), nb::arg("offsets"), nb::arg("out"))
        .def(
            "create_matmul_quant_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input, const std::vector<ScalarValuePtr> &offsets,
                TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulQuantOp(opcode, input, offsets, out));
            },
            nb::arg("opcode"), nb::arg("input"), nb::arg("offsets"), nb::arg("out"))
        .def(
            "create_ub_copy_in_op",
            [](IRBuilder &self, Opcode opcode, TensorValuePtr src, const std::vector<ScalarValuePtr> &offset,
                TileValuePtr dst) {
                return std::static_pointer_cast<Operation>(self.CreateUBCopyInOp(opcode, src, offset, dst));
            },
            nb::arg("opcode"), nb::arg("src"), nb::arg("offset"), nb::arg("dst"))
        .def(
            "create_ub_copy_out_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr src, const std::vector<ScalarValuePtr> &offset,
                TensorValuePtr dst) {
                return std::static_pointer_cast<Operation>(self.CreateUBCopyOutOp(opcode, src, offset, dst));
            },
            nb::arg("opcode"), nb::arg("src"), nb::arg("offset"), nb::arg("dst"));
}

static void IrBuilderBindMatmulOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_matmul_mmad_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulMmadOp(opcode, lhs, rhs, out));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"))
        .def(
            "create_matmul_acc_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulAccOp(opcode, lhs, rhs, out));
            },
            nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("out"));
}

static void IrBuilderBindReduceOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_reduce_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateReduceOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_reduce_with_temp_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateReduceWithTempOp(opcode, in, out, temp));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"), nb::arg("temp"));
}

static void IrBuilderBindGatherMultiInputOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_gather_in_ub_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3,
                TileValuePtr out) {
                return std::static_pointer_cast<Operation>(
                    self.CreateGatherInUBOp(opcode, input1, input2, input3, out));
            },
            nb::arg("opcode"), nb::arg("input1"), nb::arg("input2"), nb::arg("input3"), nb::arg("out"))
        .def(
            "create_gather_in_l1_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3,
                TileValuePtr out) {
                return std::static_pointer_cast<Operation>(
                    self.CreateGatherInL1Op(opcode, input1, input2, input3, out));
            },
            nb::arg("opcode"), nb::arg("input1"), nb::arg("input2"), nb::arg("input3"), nb::arg("out"));
}

static void IrBuilderBindIndexOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_index_out_cast_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr src, TileValuePtr index, TileValuePtr dst,
                TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateIndexOutCastOp(opcode, src, index, dst, out));
            },
            nb::arg("opcode"), nb::arg("src"), nb::arg("index"), nb::arg("dst"), nb::arg("out"))
        .def(
            "create_index_add_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3,
                ScalarValuePtr alpha, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(
                    self.CreateIndexAddOp(opcode, input1, input2, input3, alpha, out));
            },
            nb::arg("opcode"), nb::arg("input1"), nb::arg("input2"), nb::arg("input3"), nb::arg("alpha"),
            nb::arg("out"));
}

static void IrBuilderBindSortOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_topk_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out1, TileValuePtr out2) {
                return std::static_pointer_cast<Operation>(self.CreateTopKOp(opcode, in, out1, out2));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out1"), nb::arg("out2"))
        .def(
            "create_bitsort_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateBitSortOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_mrgsort_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMrgSortOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_extract_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateExtractOp(opcode, in, out));
            },
            nb::arg("opcode"), nb::arg("in"), nb::arg("out"))
        .def(
            "create_tiled_mrg_sort_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3,
                TileValuePtr input4, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(
                    self.CreateTiledMrgSortOp(opcode, input1, input2, input3, input4, out, temp));
            },
            nb::arg("opcode"), nb::arg("input1"), nb::arg("input2"), nb::arg("input3"), nb::arg("input4"),
            nb::arg("out"), nb::arg("temp"));
}

static void IrBuilderBindWhereTernaryOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_ternary_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr condition, TileValuePtr input, TileValuePtr other,
                TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(
                    self.CreateTernaryOp(opcode, condition, input, other, out, temp));
            },
            nb::arg("opcode"), nb::arg("condition"), nb::arg("input"), nb::arg("other"), nb::arg("out"),
            nb::arg("temp"))
        .def(
            "create_where_ts_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr condition, TileValuePtr input, ScalarValuePtr other,
                TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(
                    self.CreateWhereTSOp(opcode, condition, input, other, out, temp));
            },
            nb::arg("opcode"), nb::arg("condition"), nb::arg("input"), nb::arg("other"), nb::arg("out"),
            nb::arg("temp"))
        .def(
            "create_where_st_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr condition, ScalarValuePtr input, TileValuePtr other,
                TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(
                    self.CreateWhereSTOp(opcode, condition, input, other, out, temp));
            },
            nb::arg("opcode"), nb::arg("condition"), nb::arg("input"), nb::arg("other"), nb::arg("out"),
            nb::arg("temp"))
        .def(
            "create_where_ss_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr condition, ScalarValuePtr input, ScalarValuePtr other,
                TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(
                    self.CreateWhereSSOp(opcode, condition, input, other, out, temp));
            },
            nb::arg("opcode"), nb::arg("condition"), nb::arg("input"), nb::arg("other"), nb::arg("out"),
            nb::arg("temp"));
}

static void IrBuilderBindCompareOps(nb::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def(
            "create_compare_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr out,
                TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateCompareOp(opcode, input1, input2, out, temp));
            },
            nb::arg("opcode"), nb::arg("input1"), nb::arg("input2"), nb::arg("out"), nb::arg("temp"))
        .def(
            "create_compare_scalar_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, ScalarValuePtr input2, TileValuePtr out,
                TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(
                    self.CreateCompareScalarOp(opcode, input1, input2, out, temp));
            },
            nb::arg("opcode"), nb::arg("input1"), nb::arg("input2"), nb::arg("out"), nb::arg("temp"));
}

static void IrBuilderBindOp(nb::class_<IRBuilder> &irBuilder) {
    IrBuilderBindScalarOps(irBuilder);
    IrBuilderBindUnaryOps(irBuilder);
    IrBuilderBindBinaryOps(irBuilder);
    IrBuilderBindScalarInputOps(irBuilder);
    IrBuilderBindScalarListOps(irBuilder);
    IrBuilderBindMatmulOps(irBuilder);
    IrBuilderBindReduceOps(irBuilder);
    IrBuilderBindGatherMultiInputOps(irBuilder);
    IrBuilderBindIndexOps(irBuilder);
    IrBuilderBindSortOps(irBuilder);
    IrBuilderBindWhereTernaryOps(irBuilder);
    IrBuilderBindCompareOps(irBuilder);
}

static void IrBindBuilder(nb::module_ &m) {
    nb::class_<IRBuilderContext>(m, "IrBuilderContext")
        .def(nb::init<>())
        .def("push_scope", &IRBuilderContext::PushScope)
        .def("pop_scope", &IRBuilderContext::PopScope);

    auto irBuilder =
        nb::class_<IRBuilder>(m, "IrBuilder")
            .def(nb::init<>())
            .def("create_function", &IRBuilder::CreateFunction, nb::arg("name"), nb::arg("kind"), nb::arg("sig"))
            .def("create_tensor", &IRBuilder::CreateTensor, nb::arg("ctx"), nb::arg("shape"), nb::arg("dtype"),
                nb::arg("name") = "")
            .def("create_tile", &IRBuilder::CreateTile, nb::arg("ctx"), nb::arg("shape"), nb::arg("dtype"),
                nb::arg("name") = "")
            .def("create_scalar", &IRBuilder::CreateScalar, nb::arg("ctx"), nb::arg("dtype"), nb::arg("name") = "")
            .def(
                "create_const",
                [](IRBuilder &self, IRBuilderContext &ctx, nb::object value, const std::string &name) {
                    if (nb::isinstance<nb::int_>(value)) {
                        return self.CreateConst(ctx, nb::cast<int64_t>(value), name);
                    } else if (nb::isinstance<nb::float_>(value)) {
                        return self.CreateConst(ctx, nb::cast<double>(value), name);
                    } else {
                        throw nb::type_error("Unsupported const value type");
                    }
                },
                nb::arg("ctx"), nb::arg("value"), nb::arg("name") = "")
            .def("create_op", &IRBuilder::CreateOpStmt, nb::arg("ctx"))
            .def("create_for", &IRBuilder::CreateForStmt, nb::arg("ctx"), nb::arg("var"), nb::arg("start"),
                nb::arg("end"), nb::arg("step"))
            .def("create_if", &IRBuilder::CreateIfStmt, nb::arg("ctx"), nb::arg("cond"))
            .def("create_return", &IRBuilder::CreateReturn, nb::arg("ctx"), nb::arg("values"))
            .def(
                "enter_function",
                [](IRBuilder &self, IRBuilderContext &ctx, std::shared_ptr<Function> func) {
                    self.EnterFunctionBody(ctx, func);
                },
                nb::arg("ctx"), nb::arg("func"))
            .def(
                "enter_for",
                [](IRBuilder &self, IRBuilderContext &ctx, ForStatementPtr stmt) { self.EnterForBody(ctx, stmt); },
                nb::arg("ctx"), nb::arg("stmt"))
            .def(
                "enter_if_then",
                [](IRBuilder &self, IRBuilderContext &ctx, IfStatementPtr stmt) { self.EnterIfThen(ctx, stmt); },
                nb::arg("ctx"), nb::arg("stmt"))
            .def(
                "enter_if_else",
                [](IRBuilder &self, IRBuilderContext &ctx, IfStatementPtr stmt) { self.EnterIfElse(ctx, stmt); },
                nb::arg("ctx"), nb::arg("stmt"))
            .def(
                "exit_for",
                [](IRBuilder &self, IRBuilderContext &ctx, ForStatementPtr stmt) { self.ExitForStatement(ctx, stmt); },
                nb::arg("ctx"), nb::arg("stmt"))
            .def(
                "exit_if",
                [](IRBuilder &self, IRBuilderContext &ctx, IfStatementPtr stmt) { self.ExitIfStatement(ctx, stmt); },
                nb::arg("ctx"), nb::arg("stmt"))
            .def("emit", &IRBuilder::Emit, nb::arg("ctx"), nb::arg("operation"));

    IrBuilderBindOp(irBuilder);
}

void IrBindBlockCall(nb::module_ &m) {
    m.def(
        "call_block",
        [](const pto::FunctionPtr &blockFuncPtr, const std::vector<npu::tile_fwk::Tensor> &inputTensors, const std::vector<npu::tile_fwk::Tensor> &outputTensors,
            const std::vector<npu::tile_fwk::SymbolicScalar> &indices) {
            // 转换为 reference_wrapper
            std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> inputRefs;
            std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> outputRefs;
            for (const auto &t : inputTensors)
                inputRefs.emplace_back(t);
            for (const auto &t : outputTensors)
                outputRefs.emplace_back(t);
            // 调用 C++ 的 CallBlock (通过 blockName)
            return CallBlock(blockFuncPtr, inputRefs, outputRefs, indices);
        },
        nb::arg("block_func_ptr"), nb::arg("input_tensors"), nb::arg("output_tensors"), nb::arg("indices"),
        "Call a registered block by name (PascalCase).");
}
} // namespace pto

namespace pypto {
void BindIr(nb::module_ &m) {
    auto ir = m.def_submodule("ir", "IR module");
    pto::IrBindEnum(ir);
    pto::IrBindObjClass(ir);
    pto::IrBindType(ir);
    pto::IrBindValue(ir);
    pto::IrBindOperation(ir);
    pto::IrBindStatement(ir);
    pto::IrBindFunction(ir);
    pto::IrBindModule(ir);
    pto::IrBindBuilder(ir);
    pto::IrBindBlockCall(ir);
}
} // namespace pypto
