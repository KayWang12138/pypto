/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file operation.cpp
 * \brief
 */

#include "nb_common.h"

#include <vector>

using namespace npu::tile_fwk;

namespace pypto {
constexpr const int SCATTER_UPDATE_DIM = -2;
void BindOperation(nb::module_ &m) {
    m.def(
        "Add", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::Add(self, other); }, "Tensor add.");
    m.def(
        "Sub", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::Sub(self, other); }, "Tensor sub.");
    m.def(
        "Mul", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::Mul(self, other); }, "Tensor mul.");
    m.def(
        "Div", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::Div(self, other); }, "Tensor div.");
    m.def(
        "Fmod", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::Fmod(self, other); }, "Tensor fmod.");
    m.def(
        "BitwiseAnd", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::BitwiseAnd(self, other); }, "Tensor bitwise and.");
    m.def(
        "BitwiseOr", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::BitwiseOr(self, other); }, "Tensor bitwise or.");
    m.def(
        "BitwiseXor", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::BitwiseXor(self, other); }, "Tensor bitwise xor.");
    m.def(
        "View",
        [](const Tensor &operand, const std::vector<int64_t> &shapes, const nb::sequence &offsets) {
            bool has_symbolic = false;
            for (const auto &item : offsets) {
                if (nb::isinstance<SymbolicScalar>(item)) {
                    has_symbolic = true;
                    break;
                }
            }
            if (has_symbolic) {
                std::vector<SymbolicScalar> symbolic_offsets;
                symbolic_offsets.reserve(nb::len(offsets));
                for (const auto &item : offsets) {
                    symbolic_offsets.push_back(nb::cast<SymbolicScalar>(item));
                }
                return npu::tile_fwk::View(operand, shapes, symbolic_offsets);
            } else {
                std::vector<int64_t> int_offsets;
                int_offsets.reserve(nb::len(offsets));
                for (const auto &item : offsets) {
                    int_offsets.push_back(nb::cast<int64_t>(item));
                }
                return npu::tile_fwk::View(operand, shapes, int_offsets);
            }
        },
        nb::arg("operand"), nb::arg("shapes"), nb::arg("offsets"),
        "Create a view of a tensor. The 'offsets' can contain symbolic scalars.");
    m.def(
        "View",
        [](const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<SymbolicScalar> &newValidShapes,
            const std::vector<SymbolicScalar> &newOffsets) {
            return npu::tile_fwk::View(operand, shapes, newValidShapes, newOffsets);
        },
        nb::arg("operand"), nb::arg("shapes"), nb::arg("new_valid_shapes"), nb::arg("new_offsets"),
        "Tensor dview_pad.");
    m.def(
        "View",
        [](const Tensor &operand, const DataType dstDataType) { return npu::tile_fwk::View(operand, dstDataType); },
        nb::arg("operand"), nb::arg("dstDataType"), "Tensor view_type.");

    m.def("Exp", [](const Tensor &self) { return npu::tile_fwk::Exp(self); }, "Tensor exp.");

    m.def(
        "Transpose",
        [](const Tensor &self, const std::vector<int> &perm) { return npu::tile_fwk::Transpose(self, perm); },
        "Tensor transpose.");
    m.def("Abs", [](const Tensor &self) { return npu::tile_fwk::Abs(self); }, "Tensor abs.");
    m.def("Reciprocal", [](const Tensor &operand) { return npu::tile_fwk::Reciprocal(operand); }, "Tensor reciprocal.");
    m.def(
        "Round", [](const Tensor &self, int decimals) { return npu::tile_fwk::Round(self, decimals); }, nb::arg("self"),
        nb::arg("decimals") = 0, "Tensor round.");
    m.def("Rsqrt", [](const Tensor &self) { return npu::tile_fwk::Rsqrt(self); }, "Tensor rsqrt.");
    m.def("Sqrt", [](const Tensor &self) { return npu::tile_fwk::Sqrt(self); }, "Tensor sqrt.");
    m.def("Ceil", [](const Tensor &self) { return npu::tile_fwk::Ceil(self); }, "Tensor ceil.");
    m.def("Floor", [](const Tensor &self) { return npu::tile_fwk::Floor(self); }, "Tensor floor.");
    m.def("Trunc", [](const Tensor &self) { return npu::tile_fwk::Trunc(self); }, "Tensor trunc.");
    m.def("Reciprocal", [](const Tensor &self) { return npu::tile_fwk::Reciprocal(self); }, "Tensor Reciprocal.");
    m.def("BitwiseNot", [](const Tensor &self) { return npu::tile_fwk::BitwiseNot(self); }, "Tensor bitwisenot.");
    m.def("Neg", [](const Tensor &self) { return npu::tile_fwk::Neg(self); }, "Tensor neg.");
    m.def(
        "Log", [](const Tensor &self, const LogBaseType base) { return npu::tile_fwk::Log(self, base); },
        "Tensor log.");
    m.def(
        "Pow", [](const Tensor &self, const Element &other) { return npu::tile_fwk::Pow(self, other); }, "Tensor pow.");
    m.def(
        "Cast",
        [](const Tensor &self, DataType dstDataType, CastMode mode) {
            return npu::tile_fwk::Cast(self, dstDataType, mode);
        },
        nb::arg("operand"), nb::arg("new_data_type"), nb::arg("mode") = CAST_NONE, "Tensor cast.");

    m.def(
        "Add", [](const Tensor &self, const Element &other) { return npu::tile_fwk::Add(self, other); },
        "Tensor add scalar.");
    m.def(
        "Sub", [](const Tensor &left, const Element &right) { return npu::tile_fwk::Sub(left, right); },
        "Tensor sub scalar.");
    m.def(
        "Mul", [](const Tensor &self, const Element &other) { return npu::tile_fwk::Mul(self, other); },
        "Tensor mul scalar.");
    m.def(
        "Div", [](const Tensor &self, const Element &other) { return npu::tile_fwk::Div(self, other); },
        "Tensor div scalar.");
    m.def(
        "Fmod", [](const Tensor &self, const Element &other) { return npu::tile_fwk::Fmod(self, other); },
        "Tensor mod scalar.");
    m.def(
        "BitwiseRightShift", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::BitwiseRightShift(self, other); },
        "Tensor bitwise right shift.");
    m.def(
        "BitwiseLeftShift", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::BitwiseLeftShift(self, other); },
        "Tensor bitwise left shift.");
    m.def(
        "BitwiseRightShift", [](const Tensor &self, const Element &other) { return npu::tile_fwk::BitwiseRightShift(self, other); },
        "Tensor bitwise right shift scalar.");
    m.def(
        "BitwiseLeftShift", [](const Tensor &self, const Element &other) { return npu::tile_fwk::BitwiseLeftShift(self, other); },
        "Tensor bitwise right shift scalar.");
    m.def(
        "BitwiseRightShift", [](const Element &self, const Tensor &other) { return npu::tile_fwk::BitwiseRightShift(self, other); },
        "Scalar bitwise right shift tensor.");
    m.def(
        "BitwiseLeftShift", [](const Element &self, const Tensor &other) { return npu::tile_fwk::BitwiseLeftShift(self, other); },
        "Scalar bitwise right shift tensor.");
    m.def(
        "BitwiseAnd", [](const Tensor &self, const Element &other) { return npu::tile_fwk::BitwiseAnd(self, other); },
        "Tensor bitwiseand scalar.");
    m.def(
        "BitwiseOr", [](const Tensor &self, const Element &other) { return npu::tile_fwk::BitwiseOr(self, other); },
        "Tensor bitwiseor scalar.");
    m.def(
        "BitwiseXor", [](const Tensor &self, const Element &other) { return npu::tile_fwk::BitwiseXor(self, other); },
        "Tensor bitwisexor scalar.");
    m.def(
        "Range",
        [](const Element &start, const Element &end, const Element &step) {
            return npu::tile_fwk::Range(start, end, step);
        },
        nb::arg("start"), nb::arg("end"), nb::arg("step"), "Tensor range.");
    m.def(
        "Amax",
        [](const Tensor &operand, int axis, bool keepDim) { return npu::tile_fwk::Amax(operand, axis, keepDim); },
        nb::arg("operand"), nb::arg("axis") = -1, nb::arg("keepDim") = false, "Tensor row max single.");

    m.def(
        "Sum", [](const Tensor &operand, int axis, bool keepDim) { return npu::tile_fwk::Sum(operand, axis, keepDim); },
        nb::arg("operand"), nb::arg("axis") = -1, nb::arg("keepDim") = false, "Tensor row sum single.");
    m.def(
        "Amin",
        [](const Tensor &operand, int axis, bool keepDim) { return npu::tile_fwk::Amin(operand, axis, keepDim); },
        nb::arg("operand"), nb::arg("axis") = -1, nb::arg("keepDim") = false, "Tensor row min single.");
    m.def(
        "RowSumExpand", [](const Tensor &operand) { return npu::tile_fwk::RowSumExpand(operand); },
        "Tensor row sum expand.");
    m.def(
        "RowMaxExpand", [](const Tensor &operand) { return npu::tile_fwk::RowMaxExpand(operand); },
        "Tensor row sum expand.");
    m.def("Compact", [](const Tensor &operand) { return npu::tile_fwk::Compact(operand); }, "Tensor compact.");
    m.def(
        "IndexPut_",
        [](Tensor &self, std::vector<Tensor> indices, const Tensor &values, bool accumulate) {
            npu::tile_fwk::IndexPut_(self, indices, values, accumulate);
        },
        "Tensor indexput_.");
    m.def(
        "Scatter",
        [](const Tensor &self, const Tensor &indices, const Element &src, int axis, ScatterMode reduce) {
            return npu::tile_fwk::Scatter(self, indices, src, axis, reduce);
        },
        nb::arg("self"), nb::arg("indices"), nb::arg("src"), nb::arg("axis"), nb::arg("reduce") = ScatterMode::NONE,
        "Tensor scatter element noninplace.");
    m.def(
        "Scatter",
        [](const Tensor &self, const Tensor &indices, const Tensor &src, int axis, ScatterMode reduce) {
            return npu::tile_fwk::Scatter(self, indices, src, axis, reduce);
        },
        nb::arg("self"), nb::arg("indices"), nb::arg("src"), nb::arg("axis"), nb::arg("reduce") = ScatterMode::NONE,
        "Tensor scatter noninplace.");
    m.def(
        "IndexAdd_",
        [](const Tensor &self, const Tensor &src, const Tensor &indices, int axis, const Element &alpha) {
            return npu::tile_fwk::IndexAdd_(self, src, indices, axis, alpha);
        },
        nb::arg("self"), nb::arg("src"), nb::arg("indices"), nb::arg("axis"),
        nb::arg("alpha") = npu::tile_fwk::Element(DT_FP32, 1.0), "Tensor index add inplace.");
    m.def(
        "IndexAdd",
        [](const Tensor &self, const Tensor &src, const Tensor &indices, int axis, const Element &alpha) {
            return npu::tile_fwk::IndexAdd(self, src, indices, axis, alpha);
        },
        nb::arg("self"), nb::arg("src"), nb::arg("indices"), nb::arg("axis"),
        nb::arg("alpha") = npu::tile_fwk::Element(DT_FP32, 1.0), "Tensor index add noninplace.");
    m.def(
        "GatherElements",
        [](const Tensor &params, const Tensor &indices, int axis) {
            return npu::tile_fwk::GatherElements(params, indices, axis);
        },
        "Tensor gather element.");
    m.def(
        "Gather",
        [](const Tensor &params, const Tensor &indices, int axis) {
            return npu::tile_fwk::Gather(params, indices, axis);
        },
        "Tensor gather.");
    m.def("Duplicate", [](const Tensor &operand) { return npu::tile_fwk::Duplicate(operand); }, "Tensor duplicate.");
    m.def(
        "Full",
        [](const Element &src, DataType dType, std::vector<int64_t> dstShape, std::vector<SymbolicScalar> validShape) {
            return npu::tile_fwk::Full(src, dType, dstShape, validShape);
        },
        nb::arg("src"), nb::arg("dType"), nb::arg("dstShape"), nb::arg("validShape") = std::vector<SymbolicScalar>{},
        "Tensor vector duplicate.");
    m.def(
        "Full",
        [](const SymbolicScalar &src, DataType dType, std::vector<int64_t> dstShape,
            std::vector<SymbolicScalar> validShape) { return npu::tile_fwk::Full(src, dType, dstShape, validShape); },
        nb::arg("src"), nb::arg("dType"), nb::arg("dstShape"), nb::arg("validShape") = std::vector<SymbolicScalar>{},
        "Tensor vector duplicate.");
    m.def(
        "Load", [](const Tensor &src, const Tensor &offsets) { return npu::tile_fwk::Load(src, offsets); },
        nb::arg("src"), nb::arg("offsets"), "Tensor load.");
    m.def(
        "Reshape",
        [](const Tensor &input, const std::vector<int64_t> &dstShape, const std::vector<SymbolicScalar> validShape,
            const bool inplace) { return npu::tile_fwk::Reshape(input, dstShape, validShape, inplace); },
        nb::arg("input"), nb::arg("dstShape"), nb::arg("validShape") = std::vector<SymbolicScalar>{},
        nb::arg("inplace") = false, "Tensor reshape.");
    m.def(
        "Reshape",
        [](const Tensor &input, const std::vector<SymbolicScalar> &dstShape, const bool inplace) {
            return npu::tile_fwk::Reshape(input, dstShape, inplace);
        },
        nb::arg("input"), nb::arg("dstShape"), nb::arg("inplace"), "Tensor reshapeInplace.");
    m.def(
        "ReshapeInplace", [](const Tensor &input, Tensor &dst) { npu::tile_fwk::Reshape(input, dst); },
        nb::arg("input"), nb::arg("dst"), "Tensor reshapeInplace.");
    m.def(
        "Assign", [](const Tensor &input) { return npu::tile_fwk::Assign(input); }, nb::arg("input"), "Tensor clone.");
    m.def(
        "Reduce",
        [](const std::vector<Tensor> &aggregation, const ReduceMode &reduceMode) {
            return npu::tile_fwk::Reduce(aggregation, reduceMode);
        },
        nb::arg("aggregation"), nb::arg("reduce_mode"), "Tensor reduce.");

    m.def(
        "Maximum", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Maximum(left, right); },
        nb::arg("left"), nb::arg("right"), "Tensor maximum.");
    m.def(
        "Minimum", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Minimum(left, right); },
        nb::arg("left"), nb::arg("right"), "Tensor minimum.");
    m.def(
        "Unsqueeze",
        [](const Tensor &old, int unsqueezeDimNum) { return npu::tile_fwk::Unsqueeze(old, unsqueezeDimNum); },
        "Tensor unsqueeze.");
    m.def(
        "TensorIndex",
        [](const Tensor &params, const Tensor &indices) { return npu::tile_fwk::TensorIndex(params, indices); },
        "Tensor index.");
    m.def(
        "index_select",
        [](const Tensor &params, int dim, const Tensor &indices) {
            return npu::tile_fwk::Gather(params, indices, dim);
        },
        "Tensor index_select.");
    m.def(
        "ScatterUpdate",
        [](const Tensor &dst, const Tensor &index, const Tensor &src, int axis, std::string cacheMode, int chunkSize) {
            return npu::tile_fwk::ScatterUpdate(dst, index, src, axis, cacheMode, chunkSize);
        },
        nb::arg("dst"), nb::arg("index"), nb::arg("src"), nb::arg("axis") = SCATTER_UPDATE_DIM,
        nb::arg("cacheMode") = "PA_BNSD", nb::arg("chunkSize") = 1, "Tensor scatter update.");
    m.def(
        "Expand",
        [](const Tensor &self, const std::vector<int64_t> &dstShape, std::vector<SymbolicScalar> validShape) {
            return npu::tile_fwk::Expand(self, dstShape, validShape);
        },
        nb::arg("self"), nb::arg("dstShape"), nb::arg("validShape") = std::vector<SymbolicScalar>{}, "Tensor expand.");
    m.def(
        "NewCompact", [](const Tensor &operand) { return npu::tile_fwk::NewCompact(operand); }, "Tensor new compact.");
    m.def("LogicalNot", [](const Tensor &self) { return npu::tile_fwk::LogicalNot(self); }, "Tensor logical not.");
    m.def(
        "LogicalAnd", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::LogicalAnd(self, other); },
        "Tensor logical and.");
    m.def(
        "Where", [](const Tensor &a, const Tensor &b, const Tensor &c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def(
        "Where", [](const Tensor &a, const Tensor &b, const Element &c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def(
        "Where", [](const Tensor &a, const Element &b, const Tensor &c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def(
        "Where", [](const Tensor &a, const Element &b, const Element &c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def("Assign", [](const Tensor &operand) { return npu::tile_fwk::Assign(operand); }, "Tensor assign.");
    m.def(
        "Cat", [](const std::vector<Tensor> &tensors, int axis) { return npu::tile_fwk::Cat(tensors, axis); },
        "Tensor concat.");
    m.def("cumsum", [](const Tensor &input, int axis) { return npu::tile_fwk::CumSum(input, axis); }, "Tensor cumsum.");
    m.def(
        "TriU",
        [](const Tensor &input, const SymbolicScalar &diagonal) { return npu::tile_fwk::TriU(input, diagonal); },
        "Tensor triu.");
    m.def(
        "TriL",
        [](const Tensor &input, const SymbolicScalar &diagonal) { return npu::tile_fwk::TriL(input, diagonal); },
        "Tensor tril.");
    m.def(
        "Pad",
        [](const Tensor &old, const std::vector<int64_t> &newShape) { return npu::tile_fwk::Pad(old, newShape); },
        "Tensor pad.");
    m.def(
        "TopK",
        [](const Tensor &self, int k, int axis, bool islargest) {
            return npu::tile_fwk::TopK(self, k, axis, islargest);
        },
        nb::arg("operand"), nb::arg("k"), nb::arg("axis"), nb::arg("islargest") = true, "Tensor topk.");

    m.def(
        "Matmul",
        [](DataType out_type, const Tensor &tensor_a, const Tensor &tensor_b, bool a_trans, bool b_trans,
            bool c_matrix_nz) {
            return Matrix::Matmul(out_type, tensor_a, tensor_b, a_trans, b_trans, c_matrix_nz);
        },
        nb::arg("out_type"), nb::arg("tensor_a"), nb::arg("tensor_b"), nb::arg("a_trans") = false,
        nb::arg("b_trans") = false, nb::arg("c_matrix_nz") = false, "Matrix multiply.");

    nb::class_<Matrix::MatmulExtendParam>(m, "MatmulExtendParam")
        .def(nb::init<>())
        .def(nb::init<Tensor, Tensor, float, Matrix::ReLuType>(), nb::arg("bias_tensor"), nb::arg("scale_tensor"),
            nb::arg("scale"), nb::arg("relu_type"), "Matrix extend params.");

    m.def(
        "Matmul",
        [](DataType out_type, const Tensor &tensor_a, const Tensor &tensor_b,
            bool a_trans, bool b_trans, bool c_matrix_nz, const Matrix::MatmulExtendParam &extendParam) {
            return Matrix::Matmul(out_type, tensor_a, tensor_b, extendParam, a_trans, b_trans, c_matrix_nz);
        },
        nb::arg("out_type"), nb::arg("tensor_a"), nb::arg("tensor_b"), nb::arg("a_trans") = false,
        nb::arg("b_trans") = false, nb::arg("c_matrix_nz") = false, nb::arg("extend_params"),
        "Matrix multiply with extend param.");
    m.def(
        "BatchMatmul",
        [](DataType out_type, const Tensor &tensor_a, const Tensor &tensor_b, bool a_trans, bool b_trans,
            bool c_matrix_nz) {
            return Matrix::BatchMatmul(out_type, tensor_a, tensor_b, a_trans, b_trans, c_matrix_nz);
        },
        nb::arg("out_type"), nb::arg("a"), nb::arg("b"), nb::arg("a_trans") = false, nb::arg("b_trans") = false,
        nb::arg("c_matrix_nz") = false, "Batch matrix multiply.");
    m.def(
        "gather_in_l1",
        [](const Tensor &src, const Tensor &indices, const Tensor &blockTable, int blockSize, int size,
            bool is_b_matrix, bool is_trans) {
            if (!is_b_matrix && !is_trans) {
                std::cout << " gather in l1 m def" << std::endl;
                return experimental::GatherInL1<false, false>(src, indices, blockTable, blockSize, size);
            } else if (!is_b_matrix && is_trans) {
                return experimental::GatherInL1<false, true>(src, indices, blockTable, blockSize, size);
            } else if (is_b_matrix && !is_trans) {
                return experimental::GatherInL1<true, false>(src, indices, blockTable, blockSize, size);
            } else {
                return experimental::GatherInL1<true, true>(src, indices, blockTable, blockSize, size);
            }
        },
        nb::arg("src"), nb::arg("indices"), nb::arg("blockTable"), nb::arg("blockSize"), nb::arg("size"),
        nb::arg("is_b_matrix"), nb::arg("is_trans"), "gather load L1.");
    m.def(
        "gather_in_ub",
        [](const Tensor &param, const Tensor &indices, const Tensor &blockTable, int blockSize, int axis) {
            return experimental::GatherInUB(param, indices, blockTable, blockSize, axis);
        },
        nb::arg("param"), nb::arg("indices"), nb::arg("blockTable"), nb::arg("blockSize"), nb::arg("axis"),
        "Tensor gather_in_ub");
    m.def(
        "TransposedBatchMatmul",
        [](DataType out_type, const Tensor &tensor_a, const Tensor &tensor_b) {
            return Matrix::TransposedBatchMatmul(out_type, tensor_a, tensor_b);
        },
        nb::arg("out_type"), nb::arg("a"), nb::arg("b"), "Transposed batch matrix multiply.");
    m.def(
        "ArgSort",
        [](const Tensor &operand, int axis, bool is_largest = true) {
            return npu::tile_fwk::ArgSort(operand, axis, is_largest);
        },
        nb::arg("operand"), nb::arg("axis"), nb::arg("is_largest"), "Tensor sort.");
    m.def(
        "ScalarDivS",
        [](const Tensor &operand, const Element &value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarDivS(operand, value, reverse_operand);
        },
        nb::arg("operand"), nb::arg("value"), nb::arg("reverse_operand") = false, "Tensor scalar divs.");
    m.def(
        "ScalarAddS",
        [](const Tensor &operand, const Element &value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarAddS(operand, value, reverse_operand);
        },
        nb::arg("operand"), nb::arg("value"), nb::arg("reverse_operand") = false, "Tensor scalar adds.");
    m.def(
        "ScalarMaxS",
        [](const Tensor &operand, const Element &value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarMaxS(operand, value, reverse_operand);
        },
        nb::arg("operand"), nb::arg("value"), nb::arg("reverse_operand") = false, "Tensor scalar maxs.");
    m.def(
        "ScalarSubS",
        [](const Tensor &operand, const Element &value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarSubS(operand, value, reverse_operand);
        },
        nb::arg("operand"), nb::arg("value"), nb::arg("reverse_operand") = false, "Tensor scalar subs.");
    m.def(
        "ScalarMulS",
        [](const Tensor &operand, const Element &value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarMulS(operand, value, reverse_operand);
        },
        nb::arg("operand"), nb::arg("value"), nb::arg("reverse_operand") = false, "Tensor scalar muls.");
    m.def(
        "ScalarSub",
        [](const Tensor &operand1, const Tensor &operand2) { return npu::tile_fwk::ScalarSub(operand1, operand2); },
        nb::arg("operand1"), nb::arg("operand2"), "Tensor scalar sub.");
    m.def(
        "ScalarDiv",
        [](const Tensor &operand1, const Tensor &operand2) { return npu::tile_fwk::ScalarDiv(operand1, operand2); },
        nb::arg("operand1"), nb::arg("operand2"), "Tensor scalar div.");
    m.def(
        "Maxpool",
        [](const Tensor &operand, const std::vector<int> &pools, const std::vector<int> &stride,
            const std::vector<int> &paddings) { return npu::tile_fwk::Maxpool(operand, pools, stride, paddings); },
        nb::arg("operand"), nb::arg("pools"), nb::arg("stride"), nb::arg("paddings"), "Max pool.");
    m.def(
        "Compare",
        [](const Tensor &self, const Tensor &other, OpType op, OutType mode) {
            return npu::tile_fwk::Compare(self, other, op, mode);
        },
        nb::arg("operand1"), nb::arg("operand2"), nb::arg("operation"), nb::arg("mode"), "Tensor compare.");
    m.def(
        "Compare",
        [](const Tensor &self, const Element &other, OpType op, OutType mode) {
            return npu::tile_fwk::Compare(self, other, op, mode);
        },
        nb::arg("operand"), nb::arg("scalar"), nb::arg("operation"), nb::arg("mode"), "Tensor compare.");
    m.def(
        "Compare",
        [](const Element &self, const Tensor &other, OpType op, OutType mode) {
            return npu::tile_fwk::Compare(self, other, op, mode);
        },
        nb::arg("scalar"), nb::arg("operand"), nb::arg("operation"), nb::arg("mode"), "Tensor compare.");
    m.def(
        "Assemble",
        [](const std::vector<std::pair<Tensor, std::vector<SymbolicScalar>>> &inputs,
            Tensor &dest, bool parallel = false) {
            std::vector<npu::tile_fwk::AssembleItem> items;
            for (const auto &[tensor, offset] : inputs) {
                items.push_back({tensor, offset});
            }
            npu::tile_fwk::Assemble(items, dest, parallel);
        },
        "Tensor assemble");
    m.def(
        "Assemble",
        [](const Tensor &tensor, const std::vector<SymbolicScalar> &dynOffset, Tensor &dest) {
            npu::tile_fwk::Assemble(tensor, dynOffset, dest);
        },
        "Tensor dassemble");
    m.def("Maximum",
        [](const Tensor &operand1, const Element &operand2) { return npu::tile_fwk::Maximum(operand1, operand2); });
    m.def("Minimum",
        [](const Tensor &operand1, const Element &operand2) { return npu::tile_fwk::Minimum(operand1, operand2); });
    m.def("Clip",
        [](const Tensor &self, const Tensor &min, const Tensor &max) { return npu::tile_fwk::Clip(self, min, max); });
    m.def("Clip",
        [](const Tensor &self, const Element &min, const Element &max) { return npu::tile_fwk::Clip(self, min, max); });
    m.def(
        "OneHot", [](const Tensor &self, int numClasses) { return npu::tile_fwk::OneHot(self, numClasses); },
        "Tensor one hot.");
    m.def("PrintIf", [](SymbolicScalar cond, const std::string &format, const std::vector<Tensor> &tensors,
                         const std::vector<SymbolicScalar> &scalars) {
        npu::tile_fwk::experimental::Print(cond, format, tensors, scalars);
    });
    m.def("ToFile", [](const Tensor &operand, const std::string &fname, const std::vector<SymbolicScalar> &scalars,
                        SymbolicScalar cond) { npu::tile_fwk::ToFile(operand, fname, scalars, cond); });
    m.def(
        "topk_sort",
        [](const Tensor &x, int idx_start) {
            auto result = npu::tile_fwk::TopKSort(x, idx_start);
            // return as a Python tuple (y, temp)
            return nb::make_tuple(std::get<0>(result), std::get<1>(result));
        },
        nb::arg("x"), nb::arg("idx_start"),
        "TopKSort(x, idx_start:int) -> (y, temp)\n"
        "Performs tiled Top-K sorting starting at a scalar index.\n"
        "Returns a tuple of (sorted_values, workspace_temp)."
    );

    m.def(
        "topk_sort",
        [](const Tensor &x, const SymbolicScalar &idx_start) {
            auto result = npu::tile_fwk::TopKSort(x, idx_start);
            return nb::make_tuple(std::get<0>(result), std::get<1>(result));
        },
        nb::arg("x"), nb::arg("idx_start"),
        "TopKSort(x, idx_start:SymbolicScalar) -> (y, temp)\n"
        "Performs tiled Top-K sorting with a symbolic starting index.\n"
        "Returns a tuple of (sorted_values, workspace_temp)."
    );

    m.def(
        "topk_merge",
        [](const Tensor &x, int merge_size) {
            return npu::tile_fwk::TopKMerge(x, merge_size);
        },
        nb::arg("x"), nb::arg("merge_size"),
        "TopKMerge(x, merge_size:int) -> y\n"
        "Merges partial Top-K results into a single tensor."
    );

    m.def(
        "topk_extract",
        [](const Tensor &x, int k, bool is_index) {
            return npu::tile_fwk::TopKExtract(x, k, is_index);
        },
        nb::arg("x"), nb::arg("k"), nb::arg("is_index") = false,
        "TopKExtract(x, k:int, is_index:bool=False) -> y\n"
        "Extracts the top-k values (or indices if is_index=True)."
    );
}
} // namespace pypto
