/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file error.cpp
 * \brief Python bindings for PyPTO error classes
 */

#include "bindings.h"

#include "ir/expr.h"
#include "ir/memref.h"
#include "ir/scalar_expr.h"
#include "ir/transforms/printer.h"

namespace pypto {
namespace ir {

void BindDType(py::module& m)
{
    py::class_<ir::DataType>(m, "DataType", "Enumeration of available data types")
        .def_readonly_static("BOOL", &ir::DataType::BOOL)
        .def_readonly_static("INT4", &ir::DataType::INT4)
        .def_readonly_static("INT8", &ir::DataType::INT8)
        .def_readonly_static("INT16", &ir::DataType::INT16)
        .def_readonly_static("INT32", &ir::DataType::INT32)
        .def_readonly_static("INT64", &ir::DataType::INT64)
        .def_readonly_static("UINT4", &ir::DataType::UINT4)
        .def_readonly_static("UINT8", &ir::DataType::UINT8)
        .def_readonly_static("UINT16", &ir::DataType::UINT16)
        .def_readonly_static("UINT32", &ir::DataType::UINT32)
        .def_readonly_static("UINT64", &ir::DataType::UINT64)
        .def_readonly_static("FP4", &ir::DataType::FP4)
        .def_readonly_static("FP8E4M3FN", &ir::DataType::FP8E4M3FN)
        .def_readonly_static("FP8E5M2", &ir::DataType::FP8E5M2)
        .def_readonly_static("FP16", &ir::DataType::FP16)
        .def_readonly_static("FP32", &ir::DataType::FP32)
        .def_readonly_static("BF16", &ir::DataType::BF16)
        .def_readonly_static("HF4", &ir::DataType::HF4)
        .def_readonly_static("HF8", &ir::DataType::HF8)
        .def_readonly_static("INDEX", &ir::DataType::INDEX)
        .def("bits", &ir::DataType::GetBit, "Get the size in bits of this data type.")
        .def("c_type", &ir::DataType::ToCTypeString, "Get C style type string for code generation.")
        .def("is_float", &ir::DataType::IsFloat, "Check if this data type is a floating point type.")
        .def("is_signed", &ir::DataType::IsSignedInt, "Check if this data type is a signed integer type.")
        .def("is_unsigned", &ir::DataType::IsUnsignedInt, "Check if this data type is an unsigned integer type.")
        .def("is_int", &ir::DataType::IsInt, "Check if this data type is an integer type.")
        .def("__int__", &ir::DataType::Code, "Get the underlying type code.")
        .def("__eq__", &ir::DataType::operator==, py::arg("other"))
        .def("__ne__", &ir::DataType::operator!=, py::arg("other"))
        .def("__repr__", &ir::DataType::ToString)
        .def("__str__", &ir::DataType::ToString);
}

void BindSpan(py::module& m)
{
    py::class_<ir::Span>(m, "Span", "Source location information tracking file, line, and column positions")
        .def(
            py::init<std::string, int, int, int, int>(), py::arg("filename"), py::arg("begin_line"),
            py::arg("begin_column"), py::arg("end_line") = -1, py::arg("end_column") = -1, "Create a source span")
        .def_static("is_unknown", &ir::Span::IsUnknown, "Check if the span is unknown")
        .def_static("unknown", &ir::Span::Unknown, "Create an unknown span", py::return_value_policy::reference)
        .def("__repr__", &ir::Span::ToString)
        .def("__str__", &ir::Span::ToString)
        .def_readonly("filename", &ir::Span::filename_, "Source filename")
        .def_readonly("begin_line", &ir::Span::beginLine_, "Beginning line (1-indexed)")
        .def_readonly("begin_column", &ir::Span::beginColumn_, "Beginning column (1-indexed)")
        .def_readonly("end_line", &ir::Span::endLine_, "Ending line (1-indexed)")
        .def_readonly("end_column", &ir::Span::endColumn_, "Ending column (1-indexed)");
}

// Helper to bind a single field using reflection
template <typename ClassType, typename PyClassType, typename FieldDesc>
void BindField(PyClassType& py_class, const FieldDesc& desc)
{
    py_class.def_readonly(desc.name, desc.fieldPtr);
}

// Helper to bind all fields from a tuple of field descriptors
template <typename ClassType, typename PyClassType, typename DescTuple, std::size_t... Is>
void BindFieldsImpl(PyClassType& py_class, const DescTuple& descriptors, std::index_sequence<Is...>)
{
    (BindField<ClassType>(py_class, std::get<Is>(descriptors)), ...);
}

// Main function to bind all fields using reflection
template <typename ClassType, typename PyClassType>
void BindFields(PyClassType& py_class)
{
    constexpr auto descriptors = ClassType::GetFieldDescriptors();
    constexpr auto num_fields = std::tuple_size_v<decltype(descriptors)>;
    BindFieldsImpl<ClassType>(py_class, descriptors, std::make_index_sequence<num_fields>{});
}

void BindExpr(py::module& m)
{
    // clang-format off
    auto irnode = py::class_<IRNode>(m, "IRNode", "Base class for all IR nodes")
        .def("__str__", [](const IRNodePtr& self) {
            return PythonPrint(self, "ir");
        });

    BindFields<IRNode>(irnode);

    auto expr = py::class_<Expr, IRNode>(irnode, "Expr", "Base class for all expressions");
    BindFields<Expr>(expr);

    auto var = py::class_<Var, Expr>(expr, "Var", "Variable reference expression")
        .def(py::init<const std::string&, const TypePtr&, const Span&>(), py::arg("name"),
             py::arg("type"), py::arg("span"),
             "Create a variable reference (memory reference is stored in ShapedType for Tensor/Tile types)"
        );
    BindFields<Var>(var);

    auto iterArg = py::class_<IterArg, Var>(var, "IterArg", "Iteration argument variable")
        .def(py::init<const std::string&, const TypePtr&, const ExprPtr&, const Span&>(),
             py::arg("name"), py::arg("type"), py::arg("initValue"), py::arg("span"),
             "Create an iteration argument with initial value");
    BindFields<IterArg>(iterArg);

    auto memref = py::class_<MemRef, Var>(var, "MemRef", "Memory reference variable for shaped types (inherits from Var)")
        .def(py::init<MemorySpace, ExprPtr, uint64_t, std::string, Span>(),
             py::arg("memory_space"), py::arg("offset"), py::arg("size"), py::arg("name") = "", py::arg("span") = Span::Unknown(),
             "Create a memory reference from memory space, offset, and size")
        .def_static("same_allocation", &MemRef::SameAllocation, py::arg("a"), py::arg("b"),
                    "Check if two MemRefs share the same allocation (same base_ Ptr)")
        .def_static("may_alias", &MemRef::MayAlias, py::arg("a"), py::arg("b"),
                    "Check if two MemRefs may alias (same base + overlapping byte ranges)");
    BindFields<MemRef>(memref);

    auto constint = py::class_<ConstInt, Expr>(m, "ConstInt", "Constant integer expression")
        .def(py::init<int64_t, DataType, const Span&>(),
             py::arg("value"), py::arg("dtype"), py::arg("span"),
             "Create a constant integer expression")
        .def_property_readonly("dtype", &ConstInt::dtype, "Data type of the expression");
    BindFields<ConstInt>(constint);

    auto constfloat = py::class_<ConstFloat, Expr>(m, "ConstFloat", "Constant float expression")
        .def(py::init<double, DataType, const Span&>(),
             py::arg("value"), py::arg("dtype"), py::arg("span"),
             "Create a constant float expression")
        .def_property_readonly("dtype", &ConstFloat::dtype, "Data type of the expression");
    BindFields<ConstFloat>(constfloat);

    auto constbool_class = py::class_<ConstBool, Expr>(m, "ConstBool", "Constant boolean expression")
        .def(py::init<bool, const Span&>(),
             py::arg("value"), py::arg("span"),
             "Create a constant boolean expression")
        .def_property_readonly("dtype", &ConstBool::dtype, "Data type of the expression (always BOOL)");
    BindFields<ConstBool>(constbool_class);

    auto call = py::class_<Call, Expr>(m, "Call", "Function call expression")
        .def(py::init<const OpPtr&, const std::vector<ExprPtr>&, const Span&>(),
             py::arg("op"), py::arg("args"), py::arg("span"),
             "Create a function call expression");
    BindFields<Call>(call);

    auto make_tuple = py::class_<MakeTuple, Expr>(m, "MakeTuple", "Tuple construction expression")
        .def(py::init<const std::vector<ExprPtr>&, const Span&>(),
             py::arg("elements"), py::arg("span"),
             "Create a tuple construction expression");
    BindFields<MakeTuple>(make_tuple);

    auto tuple_get_item = py::class_<TupleGetItemExpr, Expr>(m, "TupleGetItemExpr", "Tuple element access expression")
        .def(py::init<const ExprPtr&, int, const Span&>(),
             py::arg("tuple"), py::arg("index"), py::arg("span"),
             "Create a tuple element access expression");
    BindFields<TupleGetItemExpr>(tuple_get_item);

    auto binary_expr = py::class_<BinaryExpr, Expr>(m, "BinaryExpr", "Base class for binary operations");
    BindFields<BinaryExpr>(binary_expr);

    auto unary_expr = py::class_<UnaryExpr, Expr>(m, "UnaryExpr", "Base class for unary operations");
    BindFields<UnaryExpr>(unary_expr);

#define BIND_BINARY_EXPR(OpName, Description)                                                                     \
    py::class_<OpName, BinaryExpr>(m, #OpName, Description)                                                       \
        .def(py::init<const ExprPtr&, const ExprPtr&, DataType, const Span&>(),                                   \
             py::arg("left"), py::arg("right"), py::arg("dtype"), py::arg("span"),                                \
             "Create " Description);

    // Bind all binary expression nodes
    BIND_BINARY_EXPR(Add, "Addition expression (left + right)")
    BIND_BINARY_EXPR(Sub, "Subtraction expression (left - right)")
    BIND_BINARY_EXPR(Mul, "Multiplication expression (left * right)")
    BIND_BINARY_EXPR(FloorDiv, "Floor division expression (left // right)")
    BIND_BINARY_EXPR(FloorMod, "Floor modulo expression (left % right)")
    BIND_BINARY_EXPR(FloatDiv, "Float division expression (left / right)")
    BIND_BINARY_EXPR(Min, "Minimum expression (min(left, right))")
    BIND_BINARY_EXPR(Max, "Maximum expression (max(left, right))")
    BIND_BINARY_EXPR(Pow, "Power expression (left ** right)")
    BIND_BINARY_EXPR(Eq, "Equality expression (left == right)")
    BIND_BINARY_EXPR(Ne, "Inequality expression (left != right)")
    BIND_BINARY_EXPR(Lt, "Less than expression (left < right)")
    BIND_BINARY_EXPR(Le, "Less than or equal to expression (left <= right)")
    BIND_BINARY_EXPR(Gt, "Greater than expression (left > right)")
    BIND_BINARY_EXPR(Ge, "Greater than or equal to expression (left >= right)")
    BIND_BINARY_EXPR(And, "Logical and expression (left and right)")
    BIND_BINARY_EXPR(Or, "Logical or expression (left or right)")
    BIND_BINARY_EXPR(Xor, "Logical xor expression (left xor right)")
    BIND_BINARY_EXPR(BitAnd, "Bitwise and expression (left & right)")
    BIND_BINARY_EXPR(BitOr, "Bitwise or expression (left | right)")
    BIND_BINARY_EXPR(BitXor, "Bitwise xor expression (left ^ right)")
    BIND_BINARY_EXPR(BitShiftLeft, "Bitwise left shift expression (left << right)")
    BIND_BINARY_EXPR(BitShiftRight, "Bitwise right shift expression (left >> right)")

#undef BIND_BINARY_EXPR

#define BIND_UNARY_EXPR(OpName, Description)                                                    \
    py::class_<OpName, UnaryExpr>(m, #OpName, Description)                                          \
        .def(py::init<const ExprPtr&, DataType, const Span&>(),                                     \
             py::arg("operand"), py::arg("dtype"), py::arg("span"),                                 \
             "Create " Description);

    BIND_UNARY_EXPR(Abs, "Absolute value expression (abs(operand))")
    BIND_UNARY_EXPR(Neg, "Negation expression (-operand)")
    BIND_UNARY_EXPR(Not, "Logical not expression (not operand)")
    BIND_UNARY_EXPR(BitNot, "Bitwise not expression (~operand)")
    BIND_UNARY_EXPR(Cast, "Cast expression (cast operand to dtype)")

#undef BIND_UNARY_EXPR
    // clang-format on
}

// void BindStmt(py::module& m)
// {
//     // IRNode
// }

// void BindType(py::module& m)
// {
//     // IRNode
// }
} // namespace ir

void BindIR(py::module& m)
{
    auto m1 = m.def_submodule("ir");
    ir::BindDType(m1);
    ir::BindSpan(m1);
    ir::BindExpr(m1);
}
} // namespace pypto
