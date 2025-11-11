/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file operation.cpp
 * \brief
 */

#include "pybind_common.h"

#include <vector>

using namespace npu::tile_fwk;

namespace pypto {
constexpr const int SCATTER_UPDATE_DIM  = -2;
void bind_operation(py::module &m) {
    m.def(
        "add", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::Add(self, other); }, "Tensor add.");
    m.def("sub", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::Sub(self, other); }, "Tensor sub.");
    m.def("mul", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::Mul(self, other); }, "Tensor mul.");
    m.def("div", [](const Tensor &self, const Tensor &other) { return npu::tile_fwk::Div(self, other); }, "Tensor div.");
    m.def(
        "view",
        [](const Tensor &operand, const std::vector<int64_t> &shapes, const py::sequence &offsets) {
            bool has_symbolic = false;
            for (const auto &item : offsets) {
                if (py::isinstance<SymbolicScalar>(item)) {
                    has_symbolic = true;
                    break;
                }
            }
            if (has_symbolic) {
                std::vector<SymbolicScalar> symbolic_offsets;
                symbolic_offsets.reserve(py::len(offsets));
                for (const auto &item : offsets) {
                    symbolic_offsets.push_back(item.cast<SymbolicScalar>());
                }
                return npu::tile_fwk::View(operand, shapes, symbolic_offsets);
            } else {
                std::vector<int64_t> int_offsets;
                int_offsets.reserve(py::len(offsets));
                for (const auto &item : offsets) {
                    int_offsets.push_back(item.cast<int64_t>());
                }
                return npu::tile_fwk::View(operand, shapes, int_offsets);
            }
        },
        py::arg("operand"), py::arg("shapes"), py::arg("offsets"),
        "Create a view of a tensor. The 'offsets' can contain symbolic scalars." );
    m.def(
        "view",
        [](const Tensor &operand, const std::vector<int64_t> &shapes,
            const std::vector<SymbolicScalar> &newValidShapes, const std::vector<SymbolicScalar> &newOffsets) {
            return npu::tile_fwk::View(operand, shapes, newValidShapes, newOffsets); },
        py::arg("operand"), py::arg("shapes"), py::arg("new_valid_shapes"), py::arg("new_offsets"),
        "Tensor dview_pad.");

    m.def("exp", [](const Tensor &self) { return npu::tile_fwk::Exp(self); }, "Tensor exp.");

    m.def(
        "transpose",
        [](const Tensor &self, const std::vector<int> &perm) {
            return npu::tile_fwk::Transpose(self, perm);
        },
        "Tensor transpose.");
    m.def("abs", [](const Tensor &self) { return npu::tile_fwk::Abs(self); }, "Tensor abs.");
    m.def("reciprocal", [](const Tensor &operand) { return npu::tile_fwk::Reciprocal(operand); }, "Tensor reciprocal.");
    m.def("rsqrt", [](const Tensor &self) { return npu::tile_fwk::Rsqrt(self); }, "Tensor rsqrt.");
    m.def("sqrt", [](const Tensor &self) { return npu::tile_fwk::Sqrt(self); }, "Tensor sqrt.");
    m.def("neg", [](const Tensor &self) { return npu::tile_fwk::Neg(self); }, "Tensor neg.");
    m.def("log", [](const Tensor &self, const LogBaseType base) { return npu::tile_fwk::Log(self, base); }, "Tensor log.");

    m.def(
        "cast",
        [](const Tensor &self, DataType dstDataType, CastMode mode) {
            return npu::tile_fwk::Cast(self, dstDataType, mode);
        },
        py::arg("operand"), py::arg("new_data_type"), py::arg("mode") = CAST_NONE, "Tensor cast.");

    m.def(
        "add_s", [](const Tensor &self, const Element &other) { return npu::tile_fwk::Add(self, other); },
        "Tensor add scalar.");
    m.def(
        "sub", [](const Tensor &left, const Element &right) { return npu::tile_fwk::Sub(left, right); },
        "Tensor sub scalar.");
    m.def(
        "mul_s", [](const Tensor &self, const Element &other) { return npu::tile_fwk::Mul(self, other); },
        "Tensor mul scalar.");
    m.def(
        "div_s", [](const Tensor &self, const Element &other) { return npu::tile_fwk::Div(self, other); },
        "Tensor div scalar.");
    m.def(
        "range",
        [](const Element &start, const Element &end, const Element &step) { return npu::tile_fwk::Range(start, end, step); },
        py::arg("start"), py::arg("end"), py::arg("step"), "Tensor range.");
    m.def(
        "row_max_single", [](const Tensor &operand, int axis) { return npu::tile_fwk::Amax(operand, axis); },
        py::arg("operand"), py::arg("axis") = -1, "Tensor row max single.");

    m.def(
        "row_sum_single", [](const Tensor &operand, int axis) { return npu::tile_fwk::Sum(operand, axis); },
        py::arg("operand"), py::arg("axis") = -1, "Tensor row sum single.");
    m.def(
        "row_min_single", [](const Tensor &operand, int axis) { return npu::tile_fwk::Amin(operand, axis); },
        py::arg("operand"), py::arg("axis") = -1, "Tensor row min single.");
    m.def(
        "row_sum_expand", [](const Tensor &operand) { return npu::tile_fwk::RowSumExpand(operand); },
        "Tensor row sum expand.");
    m.def(
        "row_max_expand", [](const Tensor &operand) { return npu::tile_fwk::RowMaxExpand(operand); },
        "Tensor row sum expand.");
    m.def("compact", [](const Tensor &operand) { return npu::tile_fwk::Compact(operand); }, "Tensor compact.");
    m.def("indexput", [](const Tensor &src, std::vector<Tensor> indices, const Tensor &values)
        { return npu::tile_fwk::IndexPut(src, indices, values); }, "Tensor indexput.");
    m.def("scatter_", [](const Tensor &self, const Tensor &indices, const Element &src, int axis, ScatterMode reduce)
        { return npu::tile_fwk::Scatter_(self, indices, src, axis, reduce); },
        py::arg("self"), py::arg("indices"), py::arg("src"), py::arg("axis"), py::arg("reduce") = ScatterMode::NONE,
        "Tensor scatter element inplace.");
    m.def("scatter", [](const Tensor &self, const Tensor &indices, const Element &src, int axis, ScatterMode reduce)
        { return npu::tile_fwk::Scatter(self, indices, src, axis, reduce); },
        py::arg("self"), py::arg("indices"), py::arg("src"), py::arg("axis"), py::arg("reduce") = ScatterMode::NONE,
        "Tensor scatter element noninplace.");
    m.def(
        "index_add_",
        [](const Tensor &self, const Tensor &src, const Tensor &indices, int axis, const Element &alpha) {
            return npu::tile_fwk::IndexAdd_(self, src, indices, axis, alpha);
        },
        py::arg("self"), py::arg("src"), py::arg("indices"), py::arg("axis"),
        py::arg("alpha") = npu::tile_fwk::Element(DT_FP32, 1.0), "Tensor index add inplace.");
    m.def(
        "index_add",
        [](const Tensor &self, const Tensor &src, const Tensor &indices, int axis, const Element &alpha) {
            return npu::tile_fwk::IndexAdd(self, src, indices, axis, alpha);
        },
        py::arg("self"), py::arg("src"), py::arg("indices"), py::arg("axis"),
        py::arg("alpha") = npu::tile_fwk::Element(DT_FP32, 1.0), "Tensor index add noninplace.");
    m.def("gather_element", [](const Tensor &params, const Tensor &indices, int axis)
        { return npu::tile_fwk::GatherElements(params, indices, axis); }, "Tensor gather element.");
    m.def("gather", [](const Tensor &params, const Tensor &indices, int axis)
        { return npu::tile_fwk::Gather(params, indices, axis); }, "Tensor gather.");
    m.def("duplicate", [](const Tensor &operand) { return npu::tile_fwk::Duplicate(operand); }, "Tensor duplicate.");
    m.def("full", [](const Element &src, DataType dType, std::vector<int64_t> dstShape,
        std::vector<SymbolicScalar> validShape)
        { return npu::tile_fwk::Full(src, dType, dstShape, validShape); },
        py::arg("src"), py::arg("dType"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
        "Tensor vector duplicate.");
    m.def("full", [](const SymbolicScalar &src, DataType dType, std::vector<int64_t> dstShape,
        std::vector<SymbolicScalar> validShape)
        { return npu::tile_fwk::Full(src, dType, dstShape, validShape); },
        py::arg("src"), py::arg("dType"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
        "Tensor vector duplicate.");
    m.def("reshape", [](const Tensor &input, const std::vector<int64_t> &dstShape,
        const std::vector<SymbolicScalar> validShape, const bool inplace) 
        { return npu::tile_fwk::Reshape(input, dstShape, validShape, inplace); },
        py::arg("input"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
        py::arg("inplace") = false,
        "Tensor reshape.");
    m.def("reshape", [](const Tensor &input, const std::vector<SymbolicScalar> &dstShape,
        const bool inplace) { return npu::tile_fwk::Reshape(input, dstShape, inplace); },
        py::arg("input"), py::arg("dstShape"), py::arg("inplace"),
        "Tensor reshapeInplace.");
    m.def(
        "reduce",
        [](const std::vector<Tensor> &aggregation, const ReduceMode &reduceMode) {
            return npu::tile_fwk::Reduce(aggregation, reduceMode);
        },
        py::arg("aggregation"), py::arg("reduce_mode"), "Tensor reduce.");

    m.def(
        "maximum", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Maximum(left, right); },
        py::arg("left"), py::arg("right"), "Tensor maximum.");
    m.def(
        "unsqueeze", [](const Tensor &old, int unsqueezeDimNum) { return npu::tile_fwk::Unsqueeze(old, unsqueezeDimNum); },
        "Tensor unsqueeze.");
    m.def(
        "tensor_index", [](const Tensor &params, const Tensor &indices) {
            return npu::tile_fwk::TensorIndex(params, indices);
        }, "Tensor index.");
    m.def(
        "scatter_update", [](const Tensor &dst, const Tensor &index, const Tensor &src, int axis,
        std::string cacheMode, int chunkSize) {
            return npu::tile_fwk::ScatterUpdate(dst, index, src, axis, cacheMode, chunkSize);
        },
        py::arg("dst"), py::arg("index"), py::arg("src"), py::arg("axis") = SCATTER_UPDATE_DIM, py::arg("cacheMode") = "PA_BNSD",
        py::arg("chunkSize") = 1, "Tensor scatter update.");
    m.def(
        "expand", [](const Tensor &self, const std::vector<int64_t> &dstShape, std::vector<SymbolicScalar> validShape) {
            return npu::tile_fwk::Expand(self, dstShape, validShape); },
            py::arg("self"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
            "Tensor expand.");
    m.def(
        "new_compact", [](const Tensor &operand) { return npu::tile_fwk::NewCompact(operand); },
        "Tensor new compact.");
    m.def(
        "logical_not", [](const Tensor &self) { return npu::tile_fwk::LogicalNot(self); },
        "Tensor logical not.");
    m.def(
        "where", [](const Tensor &a, const Tensor &b, const Tensor &c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def(
        "where", [](const Tensor &a, const Tensor &b, const Element &c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def(
        "where", [](const Tensor &a, const Element &b, const Tensor &c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def(
        "where", [](const Tensor &a, const Element &b, const Element &c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def(
        "assign", [](const Tensor &operand) { return npu::tile_fwk::Assign(operand); },
        "Tensor assign.");
    m.def(
        "concat", [](const std::vector<Tensor> &tensors, int axis) {
            return npu::tile_fwk::Cat(tensors, axis);
        }, "Tensor concat.");
    m.def(
        "pad", [](const Tensor &old, const std::vector<int64_t> &newShape) {
            return npu::tile_fwk::Pad(old, newShape);
        }, "Tensor pad.");
    m.def(
        "topk", [](const Tensor &self, int k, int axis, bool islargest) {
            return npu::tile_fwk::TopK(self, k, axis, islargest);
        },
        py::arg("operand"), py::arg("k"), py::arg("axis"), py::arg("islargest") = true, "Tensor topk.");

    m.def(
        "matmul",
        [](DataType out_type, const Tensor &tensor_a, const Tensor &tensor_b, bool a_trans, bool b_trans,
            bool c_matrix_nz) {
            if (!a_trans && !b_trans && !c_matrix_nz) {
                return Matrix::Matmul<false, false, false>(out_type, tensor_a, tensor_b);
            } else if (!a_trans && !b_trans && c_matrix_nz) {
                return Matrix::Matmul<false, false, true>(out_type, tensor_a, tensor_b);
            } else if (!a_trans && b_trans && !c_matrix_nz) {
                return Matrix::Matmul<false, true, false>(out_type, tensor_a, tensor_b);
            } else if (!a_trans && b_trans && c_matrix_nz) {
                return Matrix::Matmul<false, true, true>(out_type, tensor_a, tensor_b);
            } else if (a_trans && !b_trans && !c_matrix_nz) {
                return Matrix::Matmul<true, false, false>(out_type, tensor_a, tensor_b);
            } else if (a_trans && !b_trans && c_matrix_nz) {
                return Matrix::Matmul<true, false, true>(out_type, tensor_a, tensor_b);
            } else if (a_trans && b_trans && !c_matrix_nz) {
                return Matrix::Matmul<true, true, false>(out_type, tensor_a, tensor_b);
            } else {
                return Matrix::Matmul<true, true, true>(out_type, tensor_a, tensor_b);
            }
        },
        py::arg("out_type"), py::arg("tensor_a"), py::arg("tensor_b"), py::arg("a_trans") = false,
        py::arg("b_trans") = false, py::arg("c_matrix_nz") = false, "Matrix multiply.");
    m.def(
        "batch_matmul",
        [](DataType out_type, const Tensor &tensor_a, const Tensor &tensor_b, bool a_trans, bool b_trans,
            bool c_matrix_nz) {
            if (!a_trans && !b_trans && !c_matrix_nz) {
                return Matrix::BatchMatmul<false, false, false>(out_type, tensor_a, tensor_b);
            } else if (!a_trans && !b_trans && c_matrix_nz) {
                return Matrix::BatchMatmul<false, false, true>(out_type, tensor_a, tensor_b);
            } else if (!a_trans && b_trans && !c_matrix_nz) {
                return Matrix::BatchMatmul<false, true, false>(out_type, tensor_a, tensor_b);
            } else if (!a_trans && b_trans && c_matrix_nz) {
                return Matrix::BatchMatmul<false, true, true>(out_type, tensor_a, tensor_b);
            } else if (a_trans && !b_trans && !c_matrix_nz) {
                return Matrix::BatchMatmul<true, false, false>(out_type, tensor_a, tensor_b);
            } else if (a_trans && !b_trans && c_matrix_nz) {
                return Matrix::BatchMatmul<true, false, true>(out_type, tensor_a, tensor_b);
            } else if (a_trans && b_trans && !c_matrix_nz) {
                return Matrix::BatchMatmul<true, true, false>(out_type, tensor_a, tensor_b);
            } else {
                return Matrix::BatchMatmul<true, true, true>(out_type, tensor_a, tensor_b);
            }
        },
        py::arg("out_type"), py::arg("a"), py::arg("b"), py::arg("a_trans") = false, py::arg("b_trans") = false,
        py::arg("c_matrix_nz") = false, "Batch matrix multiply.");

    m.def(
        "sort",
        [](const Tensor &operand, int axis, bool is_largest = true) {
            return npu::tile_fwk::ArgSort(operand, axis, is_largest);
        },
        py::arg("operand"), py::arg("axis"), py::arg("is_largest"), "Tensor sort.");
    m.def(
        "scalar_divs",
        [](const Tensor &operand, const Element &value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarDivS(operand, value, reverse_operand);
        },
        py::arg("operand"), py::arg("value"), py::arg("reverse_operand") = false, "Tensor scalar divs.");
    m.def(
        "scalar_adds",
        [](const Tensor &operand, const Element &value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarAddS(operand, value, reverse_operand);
        },
        py::arg("operand"), py::arg("value"), py::arg("reverse_operand") = false, "Tensor scalar adds.");
    m.def(
        "scalar_maxs",
        [](const Tensor &operand, const Element &value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarMaxS(operand, value, reverse_operand);
        },
        py::arg("operand"), py::arg("value"), py::arg("reverse_operand") = false, "Tensor scalar maxs.");
    m.def(
        "scalar_subs",
        [](const Tensor &operand, const Element &value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarSubS(operand, value, reverse_operand);
        },
        py::arg("operand"), py::arg("value"), py::arg("reverse_operand") = false, "Tensor scalar subs.");
    m.def(
        "scalar_muls",
        [](const Tensor &operand, const Element &value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarMulS(operand, value, reverse_operand);
        },
        py::arg("operand"), py::arg("value"), py::arg("reverse_operand") = false, "Tensor scalar muls.");
    m.def(
        "scalar_sub",
        [](const Tensor &operand1, const Tensor &operand2) { return npu::tile_fwk::ScalarSub(operand1, operand2); },
        py::arg("operand1"), py::arg("operand2"), "Tensor scalar sub.");
    m.def(
        "scalar_div",
        [](const Tensor &operand1, const Tensor &operand2) { return npu::tile_fwk::ScalarDiv(operand1, operand2); },
        py::arg("operand1"), py::arg("operand2"), "Tensor scalar div.");
    m.def(
        "max_pool",
        [](const Tensor &operand, const std::vector<int> &pools, const std::vector<int> &stride,
            const std::vector<int> &paddings) { return npu::tile_fwk::Maxpool(operand, pools, stride, paddings); },
        py::arg("operand"), py::arg("pools"), py::arg("stride"), py::arg("paddings"), "Max pool.");
    m.def(
        "compare",
        [](const Tensor &self, const Tensor &other, OpType op, OutType mode) {
            return npu::tile_fwk::Compare(self, other, op, mode);
        },
        py::arg("operand1"), py::arg("operand2"), py::arg("operation"), py::arg("mode"), "Tensor compare.");
    m.def(
        "assemble",
        [](const std::vector<std::pair<Tensor, std::vector<int64_t>>> &tensor_int_pairs) {
            return npu::tile_fwk::Assemble(tensor_int_pairs);
        },
        "Tensor assemble");
    m.def(
        "assemble",
        [](const Tensor &tensor, const std::vector<SymbolicScalar> &dynOffset, Tensor &dest) {
            npu::tile_fwk::Assemble(tensor, dynOffset, dest);
        },
        "Tensor dassemble");
    m.def(
        "maxs",
        [](const Tensor &operand1, const Element &operand2) {
            return npu::tile_fwk::MaxS(operand1, operand2);
        }
    );
    m.def(
        "clip",
        [](const Tensor &self, const Tensor &min, const Tensor &max) {
            return npu::tile_fwk::Clip(self, min, max);
        }
    );
    m.def(
        "clip",
        [](const Tensor &self, const Element &min, const Element &max) {
            return npu::tile_fwk::Clip(self, min, max);
        }
    );
}
} // namespace pypto