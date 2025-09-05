/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>

#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Python.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

#include "operation/tilefwk_op.h"
#include "tilefwk/tensor.h"
#include "common/tile_shape.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace py = pybind11;

using namespace npu::tile_fwk;

namespace pypto {
PYBIND11_MODULE(pto, m) {
    m.doc() = "PyPTO";

    m.def("set_vec_tile_shapes", [](py::args args) {
        std::vector<int> v;
        v.reserve(args.size());
        for (auto &a : args) {
            v.push_back(a.cast<int>());
        }
        Program::GetInstance().GetTileShape().SetVecTileShapes(v);
    });
    m.def("get_vec_tile_shapes", []() { return Program::GetInstance().GetTileShape().GetVecTileShapes(); });

    m.def("add", [](Tensor left, Tensor right) { return npu::tile_fwk::Add(left, right); }, "Tensor add.");
    m.def("sub", [](Tensor left, Tensor right) { return npu::tile_fwk::Sub(left, right); }, "Tensor sub.");
    m.def("mul", [](Tensor left, Tensor right) { return npu::tile_fwk::Mul(left, right); }, "Tensor mul.");
    m.def("div", [](Tensor left, Tensor right) { return npu::tile_fwk::Div(left, right); }, "Tensor div.");
    m.def(
        "view",
        [](Tensor operand, std::vector<int> shapes, std::vector<int> offsets) {
            return npu::tile_fwk::View(operand, shapes, offsets);
        },
        "Tensor view.");
    m.def("exp", [](Tensor operand) { return npu::tile_fwk::Exp(operand); }, "Tensor exp.");

    m.def(
        "transpose",
        [](Tensor operand, std::vector<int> transposeShape) {
            return npu::tile_fwk::Transpose(operand, transposeShape);
        },
        "Tensor transpose.");
    m.def("abs", [](Tensor operand) { return npu::tile_fwk::Abs(operand); }, "Tensor abs.");
    m.def("reciprocal", [](Tensor operand) { return npu::tile_fwk::Reciprocal(operand); }, "Tensor reciprocal.");
    m.def("sqrt", [](Tensor operand) { return npu::tile_fwk::Sqrt(operand); }, "Tensor sqrt.");

    py::enum_<CastMode>(m, "CastMode")
        .value("CAST_NONE", CastMode::CAST_NONE)
        .value("CAST_RINT", CastMode::CAST_RINT)
        .value("CAST_ROUND", CastMode::CAST_ROUND)
        .value("CAST_FLOOR", CastMode::CAST_FLOOR)
        .value("CAST_CEIL", CastMode::CAST_CEIL)
        .value("CAST_TRUNC", CastMode::CAST_TRUNC)
        .value("CAST_ODD", CastMode::CAST_ODD)
        .export_values(); // export enum to Python namespace
    m.def(
        "cast",
        [](Tensor operand, DataType new_data_type, CastMode mode) {
            return npu::tile_fwk::Cast(operand, new_data_type, mode);
        },
        py::arg("operand"), py::arg("new_data_type"), py::arg("mode") = CAST_NONE, "Tensor cast.");

    m.def("add_s", [](Tensor left, Element right) { return npu::tile_fwk::AddS(left, right); }, "Tensor add scalar.");
    m.def("sub_s", [](Tensor left, Element right) { return npu::tile_fwk::SubS(left, right); }, "Tensor sub scalar.");
    m.def("mul_s", [](Tensor left, Element right) { return npu::tile_fwk::MulS(left, right); }, "Tensor mul scalar.");
    m.def("div_s", [](Tensor left, Element right) { return npu::tile_fwk::DivS(left, right); }, "Tensor div scalar.");

    m.def(
        "row_max_single", [](Tensor operand, int axis) { return npu::tile_fwk::RowMaxSingle(operand, axis); },
        py::arg("operand"), py::arg("axis") = -1, "Tensor row max single.");

    m.def(
        "row_sum_single", [](Tensor operand, int axis) { return npu::tile_fwk::RowSumSingle(operand, axis); },
        py::arg("operand"), py::arg("axis") = -1, "Tensor row sum single.");
    m.def(
        "maximum", [](Tensor left, Tensor right) { return npu::tile_fwk::Maximum(left, right); }, py::arg("left"),
        py::arg("right"), "Tensor maximum.");
    m.def(
        "rms_norm", [](Tensor operand) { return npu::tile_fwk::RmsNorm(operand); }, py::arg("operand"),
        "Tensor rms norm.");

    // cube op
    m.def(
        "set_cube_tile_shapes",
        [](std::array<int, MAX_MDIM_SIZE> m, std::vector<int> k, std::array<int, MAX_NDIM_SIZE> n,
            bool setL1Tile = false) { Program::GetInstance().GetTileShape().SetCubeTileShapes(m, k, n, setL1Tile); },
        py::arg("m"), py::arg("k"), py::arg("n"), py::arg("setL1Tile") = false);
    m.def(
        "matmul",
        [](DataType out_type, Tensor a, Tensor b, bool a_trans, bool b_trans) {
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
        [](const std::vector<std::pair<Tensor, std::vector<int>>> &tensor_int_pairs) {
            return npu::tile_fwk::Assemble(tensor_int_pairs);
        },
        "Tensor::Assemble");

    m.def("begin_function", [](const std::string &funcName, py::args args) {
        // TODO: pass function name
        std::vector<std::reference_wrapper<Tensor>> tensors;
        tensors.reserve(args.size());
        for (auto &a : args) {
            tensors.push_back(a.cast<Tensor &>());
        }
        Program::GetInstance().BeginFunction(funcName, FunctionType::STATIC, GraphType::TENSOR_GRAPH, tensors);
    });

    m.def("end_function", [](const std::string &funcName, bool generateCall) {
        Program::GetInstance().EndFunction(funcName, generateCall);
    });

    m.def("dump", []() { return Program::GetInstance().Dump(); });

    py::enum_<DataType>(m, "DataType")
        .value("DT_INT4", DataType::DT_INT4)
        .value("DT_INT8", DataType::DT_INT8)
        .value("DT_INT16", DataType::DT_INT16)
        .value("DT_INT32", DataType::DT_INT32)
        .value("DT_INT64", DataType::DT_INT64)
        .value("DT_FP8", DataType::DT_FP8)
        .value("DT_FP16", DataType::DT_FP16)
        .value("DT_FP32", DataType::DT_FP32)
        .value("DT_BF16", DataType::DT_BF16)
        .value("DT_HF4", DataType::DT_HF4)
        .value("DT_HF8", DataType::DT_HF8)
        .value("DT_UINT8", DataType::DT_UINT8)
        .value("DT_UINT16", DataType::DT_UINT16)
        .value("DT_UINT32", DataType::DT_UINT32)
        .value("DT_UINT64", DataType::DT_UINT64)
        .value("DT_BOOL", DataType::DT_BOOL)
        .value("DT_DOUBLE", DataType::DT_DOUBLE)
        .value("DT_BOTTOM", DataType::DT_BOTTOM)
        .export_values(); // export enum to Python namespace

    py::enum_<NodeType>(m, "NodeType")
        .value("LOCAL", NodeType::LOCAL)
        .value("INCAST", NodeType::INCAST)
        .value("OUTCAST", NodeType::OUTCAST)
        .export_values(); // export enum to Python namespace

    py::enum_<MemoryType>(m, "MemoryType")
        .value("MEM_UB", MemoryType::MEM_UB)
        .value("MEM_L1", MemoryType::MEM_L1)
        .value("MEM_L0A", MemoryType::MEM_L0A)
        .value("MEM_L0B", MemoryType::MEM_L0B)
        .value("MEM_L0C", MemoryType::MEM_L0C)
        .value("MEM_L2", MemoryType::MEM_L2)
        .value("MEM_L3", MemoryType::MEM_L3)
        .value("MEM_DEVICE_DDR", MemoryType::MEM_DEVICE_DDR)
        .value("MEM_HOST1", MemoryType::MEM_HOST1)
        .value("MEM_FAR1", MemoryType::MEM_FAR1)
        .value("MEM_FAR2", MemoryType::MEM_FAR2)
        .value("MEM_UNKNOWN", MemoryType::MEM_UNKNOWN)
        .export_values(); // export enum to Python namespace

    py::enum_<GraphType>(m, "graph_type")
        .value("TENSOR_GRAPH", GraphType::TENSOR_GRAPH)
        .value("TILE_GRAPH", GraphType::TILE_GRAPH)
        .value("ROOT_GRAPH", GraphType::ROOT_GRAPH)
        .value("LEAF_GRAPH", GraphType::LEAF_GRAPH)
        .value("LEAF_VF_GRAPH", GraphType::LEAF_VF_GRAPH)
        .value("INVALID", GraphType::INVALID)
        .export_values(); // export enum to Python namespace

    py::class_<Tensor>(m, "tensor")
        .def(py::init<DataType, std::vector<int>, std::string>(), py::arg("dtype"), py::arg("shape"),
            py::arg("name") = "Unknown")
        .def(
            "__add__", [](Tensor &self, Tensor tensor) { return npu::tile_fwk::Add(self, tensor); }, "Tensor add.")
        .def("get_dtype", &Tensor::GetDataType)
        .def_property_readonly(
            "shape", py::overload_cast<>(&Tensor::GetShape, py::const_), py::return_value_policy::reference_internal)
        .def("get_shape", py::overload_cast<>(&Tensor::GetShape, py::const_),
            py::return_value_policy::reference_internal);

    py::class_<Element>(m, "element")
        .def(py::init<DataType, int64_t>(), py::arg("type"), py::arg("sData"))
        .def(py::init<DataType, uint64_t>(), py::arg("type"), py::arg("uData"))
        .def(py::init<DataType, double>(), py::arg("type"), py::arg("fData"))
        .def("get_data_type", &Element::GetDataType)
        .def("get_signed_data", &Element::GetSignedData)
        .def("get_unsigned_data", &Element::GetUnsignedData)
        .def("get_float_data", &Element::GetFloatData);

    // TODO(anastasios): not used now since we use Program::GetInstance()
    py::class_<TileShape>(m, "tile_shape")
        .def(py::init<>())
        .def("reset", &TileShape::Reset)
        .def("dump", &TileShape::Dump)
        .def("get_vec_tile_shapes", &TileShape::GetVecTileShapes)
        .def("specify_static_rank_id", &TileShape::SpecifyStaticRankId)
        //.def("SetVecTileShapes", py::overload_cast<const std::vector<int> &>(&TileShape::SetVecTileShapes))
        // Bind a Python *args wrapper for the variadic template
        .def("set_vec_tile_shapes", [](TileShape &self, py::args args) {
            std::vector<int> v;
            v.reserve(args.size());
            for (auto &a : args) {
                v.push_back(a.cast<int>()); // require ints
            }
            self.SetVecTileShapes(v);
        });
};
} // namespace pypto
