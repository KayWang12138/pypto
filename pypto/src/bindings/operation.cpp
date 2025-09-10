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
void bind_operation(py::module &m) {
    m.def(
        "add", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Add(left, right); }, "Tensor add.");
    m.def("sub", [](const Tensor left, const Tensor &right) { return npu::tile_fwk::Sub(left, right); }, "Tensor sub.");
    m.def("mul", [](const Tensor left, const Tensor &right) { return npu::tile_fwk::Mul(left, right); }, "Tensor mul.");
    m.def("div", [](const Tensor left, const Tensor &right) { return npu::tile_fwk::Div(left, right); }, "Tensor div.");
    m.def(
        "view",
        [](const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<int64_t> &offsets) {
            return npu::tile_fwk::View(operand, shapes, offsets);
        },
        "Tensor view.");
    m.def("exp", [](const Tensor &operand) { return npu::tile_fwk::Exp(operand); }, "Tensor exp.");

    m.def(
        "transpose",
        [](const Tensor &operand, const std::vector<int> &transposeShape) {
            return npu::tile_fwk::Transpose(operand, transposeShape);
        },
        "Tensor transpose.");
    m.def("abs", [](const Tensor &operand) { return npu::tile_fwk::Abs(operand); }, "Tensor abs.");
    m.def("reciprocal", [](const Tensor &operand) { return npu::tile_fwk::Reciprocal(operand); }, "Tensor reciprocal.");
    m.def("sqrt", [](const Tensor &operand) { return npu::tile_fwk::Sqrt(operand); }, "Tensor sqrt.");

    m.def(
        "cast",
        [](const Tensor &operand, DataType new_data_type, CastMode mode) {
            return npu::tile_fwk::Cast(operand, new_data_type, mode);
        },
        py::arg("operand"), py::arg("new_data_type"), py::arg("mode") = CAST_NONE, "Tensor cast.");

    m.def(
        "add_s", [](const Tensor &left, const Element &right) { return npu::tile_fwk::AddS(left, right); },
        "Tensor add scalar.");
    m.def(
        "sub_s", [](const Tensor &left, const Element &right) { return npu::tile_fwk::SubS(left, right); },
        "Tensor sub scalar.");
    m.def(
        "mul_s", [](const Tensor &left, const Element &right) { return npu::tile_fwk::MulS(left, right); },
        "Tensor mul scalar.");
    m.def(
        "div_s", [](const Tensor &left, const Element &right) { return npu::tile_fwk::DivS(left, right); },
        "Tensor div scalar.");

    m.def(
        "row_max_single", [](const Tensor &operand, int axis) { return npu::tile_fwk::RowMaxSingle(operand, axis); },
        py::arg("operand"), py::arg("axis") = -1, "Tensor row max single.");

    m.def(
        "row_sum_single", [](const Tensor &operand, int axis) { return npu::tile_fwk::RowSumSingle(operand, axis); },
        py::arg("operand"), py::arg("axis") = -1, "Tensor row sum single.");
    m.def(
        "row_min_single", [](const Tensor &operand, int axis) { return npu::tile_fwk::RowMinSingle(operand, axis); },
        py::arg("operand"), py::arg("axis") = -1, "Tensor row min single.");
    m.def(
        "row_sum_expand", [](const Tensor &operand) { return npu::tile_fwk::RowSumExpand(operand); },
        "Tensor row sum expand.");
    m.def(
        "row_max_expand", [](const Tensor &operand) { return npu::tile_fwk::RowSumSingle(operand); },
        "Tensor row sum expand.");
    m.def("compact", [](const Tensor &operand) { return npu::tile_fwk::Compact(operand); }, "Tensor compact.");
    m.def("indexput", [](const Tensor &src, std::vector<Tensor> indices, const Tensor &values)
        { return npu::tile_fwk::IndexPut(src, indices, values); }, "Tensor indexput.");
    m.def("scatter_element", [](const Tensor &src, const Tensor &idx, const Element &scalar, int axis)
        { return npu::tile_fwk::ScatterElement(src, idx, scalar, axis); }, "Tensor scatter element.");
    m.def("gather_element", [](const Tensor &params, const Tensor &indices, int axis)
        { return npu::tile_fwk::GatherElement(params, indices, axis); }, "Tensor gather element.");
    m.def("gather", [](const Tensor &params, const Tensor &indices, int axis)
        { return npu::tile_fwk::Gather(params, indices, axis); }, "Tensor gather.");
    m.def("duplicate", [](const Tensor &operand) { return npu::tile_fwk::Duplicate(operand); }, "Tensor duplicate.");
    m.def("vector_duplicate", [](const Element &src, DataType dType, std::vector<int64_t> dstShape, 
        std::vector<SymbolicScalar> validShape)
        { return npu::tile_fwk::VectorDuplicate(src, dType, dstShape, validShape); },
        py::arg("src"), py::arg("dType"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
        "Tensor vector duplicate.");
    m.def("vector_duplicate", [](const SymbolicScalar &src, DataType dType, std::vector<int64_t> dstShape, 
        std::vector<SymbolicScalar> validShape)
        { return npu::tile_fwk::VectorDuplicate(src, dType, dstShape, validShape); },
        py::arg("src"), py::arg("dType"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
        "Tensor vector duplicate.");
    m.def("reshape", [](const Tensor &input, const std::vector<int64_t> &dstShape,
        const std::vector<SymbolicScalar> validShape) { return npu::tile_fwk::Reshape(input, dstShape, validShape); },
        py::arg("input"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
        "Tensor reshape.");

    m.def(
        "maximum", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Maximum(left, right); },
        py::arg("left"), py::arg("right"), "Tensor maximum.");
    m.def(
        "rms_norm", [](const Tensor &operand) { return npu::tile_fwk::RmsNorm(operand); }, py::arg("operand"),
        "Tensor rms norm.");

    m.def(
        "matmul",
        [](DataType out_type, const Tensor &a, const Tensor &b, bool a_trans, bool b_trans) {
            if (a_trans) {
                if (b_trans) {
                    return npu::tile_fwk::Matrix::Matmul<true, true>(out_type, a, b);
                } else {
                    return npu::tile_fwk::Matrix::Matmul<true, false>(out_type, a, b);
                }
            } else {
                if (b_trans) {
                    return npu::tile_fwk::Matrix::Matmul<false, true>(out_type, a, b);
                } else {
                    return npu::tile_fwk::Matrix::Matmul<false, false>(out_type, a, b);
                }
            }
        },
        py::arg("out_type"), py::arg("a"), py::arg("b"), py::arg("a_trans") = false, py::arg("b_trans") = false,
        "Matrix multiply.");

    m.def(
        "assemble",
        [](const std::vector<std::pair<Tensor, std::vector<int64_t>>> &tensor_int_pairs) {
            return npu::tile_fwk::Assemble(tensor_int_pairs);
        },
        "Tensor::Assemble");
}
} // namespace pypto