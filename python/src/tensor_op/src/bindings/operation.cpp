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
        "add", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Add(left, right); }, "Tensor add.");
    m.def("sub", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Sub(left, right); }, "Tensor sub.");
    m.def("mul", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Mul(left, right); }, "Tensor mul.");
    m.def("div", [](const Tensor &left, const Tensor &right) { return npu::tile_fwk::Div(left, right); }, "Tensor div.");
    m.def(
        "View",
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
        "View",
        [](const Tensor &operand, const std::vector<int64_t> &shapes,
            const std::vector<SymbolicScalar> &newValidShapes, const std::vector<SymbolicScalar> &newOffsets) {
            return npu::tile_fwk::View(operand, shapes, newValidShapes, newOffsets); },
        py::arg("operand"), py::arg("shapes"), py::arg("new_valid_shapes"), py::arg("new_offsets"),
        "Tensor dview_pad.");

    m.def("exp", [](const Tensor &operand) { return npu::tile_fwk::Exp(operand); }, "Tensor exp.");

    m.def(
        "transpose",
        [](const Tensor &operand, const std::vector<int> &transposeShape) {
            return npu::tile_fwk::Transpose(operand, transposeShape);
        },
        "Tensor transpose.");
    m.def("abs", [](const Tensor &operand) { return npu::tile_fwk::Abs(operand); }, "Tensor abs.");
    m.def("reciprocal", [](const Tensor &operand) { return npu::tile_fwk::Reciprocal(operand); }, "Tensor reciprocal.");
    m.def("rsqrt", [](const Tensor &operand) { return npu::tile_fwk::Rsqrt(operand); }, "Tensor rsqrt.");
    m.def("sqrt", [](const Tensor &operand) { return npu::tile_fwk::Sqrt(operand); }, "Tensor sqrt.");
    m.def("neg", [](const Tensor &operand) { return npu::tile_fwk::Neg(operand); }, "Tensor neg.");
    m.def("log", [](const Tensor &operand, const LogBaseType base) { return npu::tile_fwk::Log(operand, base); }, "Tensor log.");

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
        "range",
        [](const Element &start, const Element &end, const Element &step) { return npu::tile_fwk::Range(start, end, step); },
        py::arg("start"), py::arg("end"), py::arg("step"), "Tensor range.");
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
    m.def("scatter_", [](const Tensor &self, const Tensor &indices, const Element &src, int axis, std::string reduce)
        { return npu::tile_fwk::Scatter_(self, indices, src, axis, reduce); },
        py::arg("self"), py::arg("indices"), py::arg("src"), py::arg("axis"), py::arg("reduce") = "",
        "Tensor scatter element inplace.");
    m.def("scatter", [](const Tensor &self, const Tensor &indices, const Element &src, int axis, std::string reduce)
        { return npu::tile_fwk::Scatter(self, indices, src, axis, reduce); },
        py::arg("self"), py::arg("indices"), py::arg("src"), py::arg("axis"), py::arg("reduce") = "",
        "Tensor scatter element noninplace.");
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
    m.def("Reshape", [](const Tensor &input, const std::vector<int64_t> &dstShape,
        const std::vector<SymbolicScalar> validShape) { return npu::tile_fwk::Reshape(input, dstShape, validShape); },
        py::arg("input"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
        "Tensor reshape.");
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
        "sin", [](const Tensor &operand) { return npu::tile_fwk::Sin(operand); },
        "Tensor sin.");
    m.def(
        "cos", [](const Tensor &operand) { return npu::tile_fwk::Cos(operand); },
        "Tensor cos.");
    m.def(
        "new_compact", [](const Tensor &operand) { return npu::tile_fwk::NewCompact(operand); },
        "Tensor new compact.");
    m.def(
        "logical_not", [](const Tensor &operand) { return npu::tile_fwk::LogicalNot(operand); },
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
        "rms_norm", [](const Tensor &operand) { return npu::tile_fwk::RmsNorm(operand); }, py::arg("operand"),
        "Tensor rms norm.");
    m.def(
        "rms_norm", [](const Tensor &operand, const Tensor &gamma, float epsilon) {
            return npu::tile_fwk::RmsNorm(operand, gamma, epsilon);
        },
        py::arg("operand"), py::arg("gamma"), py::arg("epsilon") = 1e-05f, "Tensor rms norm.");
    m.def(
        "concat", [](const std::vector<Tensor> &tensorLists, int axis) {
            return npu::tile_fwk::Concat(tensorLists, axis);
        }, "Tensor concat.");
    m.def(
        "pad", [](const Tensor &old, const std::vector<int64_t> &newShape) {
            return npu::tile_fwk::Pad(old, newShape);
        }, "Tensor pad.");
    m.def(
        "topk", [](const Tensor &operand, const int &k, int axis, bool islargest) {
            return npu::tile_fwk::TopK(operand, k, axis, islargest);
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
        "softmax", [](const Tensor &operand) { return npu::tile_fwk::SoftmaxNew(operand); }, py::arg("operand"),
        "Tensor softmax.");
    m.def(
        "rotate_half", [](const Tensor &input) { return npu::tile_fwk::RotateHalf(input); }, py::arg("input"),
        "Tensor rotate half.");
    m.def("sigmoid", [](Tensor &input) { return npu::tile_fwk::Sigmoid(input); }, py::arg("input"), "Tensor sigmoid.");
    m.def(
        "quant",
        [](const Tensor &input, bool is_symmetry = true, bool has_smooth_factor = false,
            const py::object &smooth_factor = py::none()) {
            Tensor smooth_factor_tensor = smooth_factor.is_none() ? Tensor() : smooth_factor.cast<Tensor>();
            return npu::tile_fwk::Quant(input, is_symmetry, has_smooth_factor, smooth_factor_tensor);
        },
        py::arg("input"), py::arg("is_symmetry") = true, py::arg("has_smooth_factor") = false,
        py::arg("smooth_factor") = py::none(), "Tensor quant.");
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
        "reduce",
        [](const std::vector<Tensor> &aggregation, const ReduceMode reduce_mode) {
            return npu::tile_fwk::Reduce(aggregation, reduce_mode);
        },
        py::arg("aggregation"), py::arg("reduce_mode"), "Tensor reduce.");
    m.def(
        "max_pool",
        [](const Tensor &operand, const std::vector<int> &pools, const std::vector<int> &stride,
            const std::vector<int> &paddings) { return npu::tile_fwk::Maxpool(operand, pools, stride, paddings); },
        py::arg("operand"), py::arg("pools"), py::arg("stride"), py::arg("paddings"), "Max pool.");
    py::class_<RoPETileShapeConfig>(m, "rope_tile_shape_config")
        .def(py::init<>())
        .def_readwrite("two_dims_tile_shape", &RoPETileShapeConfig::twoDimsTileShape)
        .def_readwrite("three_dims_tile_shape", &RoPETileShapeConfig::threeDimsTileShape)
        .def_readwrite("four_dims_tile_shape", &RoPETileShapeConfig::fourDimsTileShape)
        .def_readwrite("five_dims_tile_shape", &RoPETileShapeConfig::fiveDimsTileShape);
    py::class_<PaTileShapeConfig>(m, "pa_tile_shape_config")
        .def(py::init<>())
        .def_readwrite("head_num_q_tile", &PaTileShapeConfig::headNumQTile)
        .def_readwrite("v0_tile_shape", &PaTileShapeConfig::v0TileShape)
        .def_readwrite("c1_tile_shape", &PaTileShapeConfig::c1TileShape)
        .def_readwrite("v1_tile_shape", &PaTileShapeConfig::v1TileShape)
        .def_readwrite("c2_tile_shape", &PaTileShapeConfig::c2TileShape)
        .def_readwrite("v2_tile_shape", &PaTileShapeConfig::v2TileShape);
    m.def(
        "apply_rotary_pos_emb",
        [](const Tensor &q, const Tensor &k, const Tensor &cos, const Tensor &sin, const Tensor &position_ids,
            Tensor &q_embed, Tensor &k_embed, const int unsqueeze_dim = 1,
            const py::object &rope_tile_config = py::none()) {
            auto config =
                rope_tile_config.is_none() ? RoPETileShapeConfig() : rope_tile_config.cast<RoPETileShapeConfig>();
            npu::tile_fwk::ApplyRotaryPosEmb(q, k, cos, sin, position_ids, q_embed, k_embed, unsqueeze_dim, config);
        },
        py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"), py::arg("position_ids"), py::arg("q_embed"),
        py::arg("k_embed"), py::arg("unsqueeze_dim") = 1, py::arg("rope_tile_config") = py::none(),
        "Apply rotary pos emb.");
    py::class_<RoPETileShapeConfigNew>(m, "rope_tile_shape_config_new")
        .def(py::init<>())
        .def_readwrite("three_dims_tile_shape", &RoPETileShapeConfigNew::threeDimsTileShape)
        .def_readwrite("four_dims_tile_shape_q", &RoPETileShapeConfigNew::fourDimsTileShapeQ)
        .def_readwrite("four_dims_tile_shape_k", &RoPETileShapeConfigNew::fourDimsTileShapeK)
        .def_readwrite("five_dims_tile_shape", &RoPETileShapeConfigNew::fiveDimsTileShape);
    m.def(
        "apply_rotary_pos_emb_v2",
        [](const Tensor &q, const Tensor &k, const Tensor &cos, const Tensor &sin, Tensor &q_embed, Tensor &k_embed,
            const int unsqueeze_dim = NUM2, const py::object &rope_tile_config = py::none()) {
            auto config =
                rope_tile_config.is_none() ? RoPETileShapeConfigNew() : rope_tile_config.cast<RoPETileShapeConfigNew>();
            npu::tile_fwk::ApplyRotaryPosEmbV2(q, k, cos, sin, q_embed, k_embed, unsqueeze_dim, config);
        },
        py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"), py::arg("q_embed"), py::arg("k_embed"),
        py::arg("unsqueeze_dim") = NUM2, py::arg("rope_tile_config") = py::none(), "Apply rotary pos emb v2.");
    m.def(
        "incre_flash_attention",
        [](Tensor &q_nope, Tensor &k_nope_cache, Tensor &v_nope_cache, Tensor &q_rope, Tensor &k_rope_cache,
            std::vector<std::vector<int>> &block_table, std::vector<int> &act_seqs, float soft_max_scale,
            Tensor &attention_out, IfaTileShapeConfig &tile_config) {
            npu::tile_fwk::IncreFlashAttention(q_nope, k_nope_cache, v_nope_cache, q_rope, k_rope_cache, block_table,
                act_seqs, soft_max_scale, attention_out, tile_config);
        },
        py::arg("q_nope"), py::arg("k_nope_cache"), py::arg("v_nope_cache"), py::arg("q_rope"), py::arg("k_rope_cache"),
        py::arg("block_table"), py::arg("act_seqs"), py::arg("soft_max_scale"), py::arg("attention_out"),
        py::arg("tile_config"), "Incre flash attention.");
    m.def(
        "page_attention_adds",
        [](Tensor &q_nope, Tensor &k_nope_cache, Tensor &v_nope_cache, Tensor &q_rope, Tensor &k_rope_cache,
            Tensor &block_table, Tensor &act_seqs, int block_size, float soft_max_scale, Tensor &attention_out,
            Tensor &post_out, PaTileShapeConfig &tile_config, int max_unroll_times = 1) {
            npu::tile_fwk::PageAttentionAddS(q_nope, k_nope_cache, v_nope_cache, q_rope, k_rope_cache, block_table,
                act_seqs, block_size, soft_max_scale, attention_out, post_out, tile_config, max_unroll_times);
        },
        py::arg("q_nope"), py::arg("k_nope_cache"), py::arg("v_nope_cache"), py::arg("q_rope"), py::arg("k_rope_cache"),
        py::arg("block_table"), py::arg("act_seqs"), py::arg("block_size"), py::arg("soft_max_scale"),
        py::arg("attention_out"), py::arg("post_out"), py::arg("tile_config"), py::arg("max_unroll_times") = 1,
        "Page attention adds.");
    m.def(
        "page_attention_adds_single_output",
        [](Tensor &q_nope, Tensor &k_nope_cache, Tensor &v_nope_cache, Tensor &q_rope, Tensor &k_rope_cache,
            Tensor &block_table, Tensor &act_seqs, int block_size, float soft_max_scale, Tensor &attention_out,
            Tensor &post_out, PaTileShapeConfig &tile_config, int max_unroll_times = 1) {
            npu::tile_fwk::PageAttentionAddSSingleOutput(q_nope, k_nope_cache, v_nope_cache, q_rope, k_rope_cache,
                block_table, act_seqs, block_size, soft_max_scale, attention_out, post_out, tile_config,
                max_unroll_times);
        },
        py::arg("q_nope"), py::arg("k_nope_cache"), py::arg("v_nope_cache"), py::arg("q_rope"), py::arg("k_rope_cache"),
        py::arg("block_table"), py::arg("act_seqs"), py::arg("block_size"), py::arg("soft_max_scale"),
        py::arg("attention_out"), py::arg("post_out"), py::arg("tile_config"), py::arg("max_unroll_times") = 1,
        "Page attention adds single output.");
    m.def(
        "prolog_post",
        [](Tensor &q_nope, Tensor &k_nope_cache, Tensor &v_nope_cache, Tensor &q_rope, Tensor &k_rope_cache,
            Tensor &block_table, Tensor &act_seqs, Tensor &weight_uv, Tensor &weight0, int block_size,
            float soft_max_scale, Tensor &post_out, PaTileShapeConfig &tile_config) {
        npu::tile_fwk::PrologPost(q_nope, k_nope_cache, v_nope_cache, q_rope, k_rope_cache, block_table, act_seqs,
            weight_uv, weight0, block_size, soft_max_scale, post_out, tile_config);
        },
        py::arg("q_nope"), py::arg("k_nope_cache"), py::arg("v_nope_cache"), py::arg("q_rope"), py::arg("k_rope_cache"),
        py::arg("block_table"), py::arg("act_seqs"), py::arg("weight_uv"), py::arg("weight0"), py::arg("block_size"),
        py::arg("soft_max_scale"), py::arg("post_out"), py::arg("tile_config"), "Prolog post.");
    m.def(
        "all_gather",
        [](const Tensor &in, std::vector<Tensor> &out, const char *group) {
        npu::tile_fwk::Distributed::AllGather(in, out, group);
        },
        py::arg("in"), py::arg("out"), py::arg("group"), "Tensor all gather.");
    m.def(
        "all_gather",
        [](const Tensor &in, const char *group) {
        return npu::tile_fwk::Distributed::AllGather(in, group); },
        py::arg("in"), py::arg("group"), "Tensor all gather.");
    m.def(
        "reduce_scatter",
        [](const std::vector<Tensor> &in, const char *group, Distributed::DistReduceType reduce_type) {
            return npu::tile_fwk::Distributed::ReduceScatter(in, group, reduce_type);
        },
        py::arg("in"), py::arg("group"), py::arg("reduce_type"), "Tensor reduce scatter.");
    m.def(
        "reduce_scatter",
        [](const Tensor &in, const char *group, Distributed::DistReduceType reduce_type) {
            return npu::tile_fwk::Distributed::ReduceScatter(in, group, reduce_type);
        },
        py::arg("in"), py::arg("group"), py::arg("reduce_type"), "Tensor reduce scatter.");
    m.def(
        "moe_dispatch",
        [](const Tensor &token_tensor, const Tensor &token_expert_table, Tensor &valid_cnt, const char *group) {
            return npu::tile_fwk::Distributed::MoeDispatch(token_tensor, token_expert_table, valid_cnt, group);
        },
        py::arg("token_tensor"), py::arg("token_expert_table"), py::arg("valid_cnt"), py::arg("group"),
        "Tensor moe dispatch.");
    m.def(
        "moe_combine",
        [](const Tensor &in, const Tensor &scale, const Tensor &combine_info, const char *group) {
            return npu::tile_fwk::Distributed::MoeCombine(in, scale, combine_info, group);
        },
        py::arg("in"), py::arg("scale"), py::arg("combine_info"), py::arg("group"), "Tensor moe combine.");

    m.def(
        "Assemble",
        [](const std::vector<std::pair<Tensor, std::vector<int64_t>>> &tensor_int_pairs) {
            return npu::tile_fwk::Assemble(tensor_int_pairs);
        },
        "Tensor assemble");
    m.def(
        "Assemble",
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
}
} // namespace pypto