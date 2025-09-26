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
 * \file controller.cpp
 * \brief
 */

#include "pybind_common.h"

#include <utility>
#include <vector>

using namespace npu::tile_fwk;
using ref_tensors = std::vector<std::reference_wrapper<const Tensor>>;

using ref_tensors = std::vector<std::reference_wrapper<const Tensor>>;

namespace pypto {
void bind_controller(py::module &m) {
    m.def("set_vec_tile_shapes", [](py::args args) {
        std::vector<int64_t> v;
        v.reserve(args.size());
        for (auto &a : args) {
            v.push_back(a.cast<int64_t>());
        }
        TileShape::Current().SetVecTile(v);
    });
    m.def("get_vec_tile_shapes", []() { return TileShape::Current().GetVecTile(); });

    m.def("set_cube_tile_shapes",
        [](const std::vector<int64_t>& mvec, const std::vector<int64_t>& kvec,
        const std::vector<int64_t>& nvec, bool setL1Tile = false) {
            if (mvec.size() > MAX_M_DIM_SIZE) {
                throw py::value_error("Parameter 'm' must have exactly " +
                                    std::to_string(MAX_M_DIM_SIZE) + " elements");
            }
            if (kvec.size() > MAX_K_DIM_SIZE) {
                throw py::value_error("Parameter 'k' must have exactly " +
                                    std::to_string(MAX_K_DIM_SIZE) + " elements");
            }
            if (nvec.size() > MAX_N_DIM_SIZE) {
                throw py::value_error("Parameter 'n' must have exactly " +
                                    std::to_string(MAX_N_DIM_SIZE) + " elements");
            }

            std::array<int64_t, MAX_M_DIM_SIZE> marr = {0};
            std::array<int64_t, MAX_K_DIM_SIZE> karr = {0};
            std::array<int64_t, MAX_N_DIM_SIZE> narr = {0};

            std::copy(mvec.begin(), mvec.end(), marr.begin());
            std::copy(kvec.begin(), kvec.end(), karr.begin());
            std::copy(nvec.begin(), nvec.end(), narr.begin());

            TileShape::Current().SetCubeTile(marr, karr, narr, setL1Tile);
        },
        py::arg("m"), py::arg("k"), py::arg("n"), py::arg("setL1Tile") = false,
        "Set cube tile shapes with specified dimensions");

    m.def(
        "set_config", 
        [](const std::string &key, const int &value) {
            Program::GetInstance().GetConfig().Set<int>(key, value); }, 
        py::arg("key"), py::arg("value"));
    m.def(
        "set_config", 
        [](const std::string &key, const std::map<int, int> &value) { 
            Program::GetInstance().GetConfig().Set<std::map<int, int>>(key, value); }, 
        py::arg("key"), py::arg("value"));
    m.def(
        "set_matrix_size", 
        [](const std::vector<int64_t>& size) {
            TileShape::Current().SetMatrixSize(size); }, 
        py::arg("size"));
    m.def(
        "set_semantic_label", 
        [](const std::string &label) {
            ConfigManager::Instance().SetSemanticLabel(label); }, 
        py::arg("label"));

    m.def(
        "set_operation_config", 
        [](const std::string &key, const bool &value) {
            ConfigManager::Instance().SetOperationConfig<bool>(key, value); }, 
        py::arg("key"), py::arg("value"));
    m.def(
        "set_pass_config", 
        [](const std::string &strategy, const std::string &identifier, const std::string &key, const bool &value) {
            ConfigManager::Instance().SetPassConfig<bool>(strategy, identifier, key, value); }, 
        py::arg("strategy"), py::arg("identifier"), py::arg("key"), py::arg("value"));
    m.def(
        "set_host_config", 
        [](const std::string &key, const bool &value) {
            ConfigManager::Instance().SetHostConfig<bool>(key, value); }, 
        py::arg("key"), py::arg("value"));
    m.def("bytes_of", [](DataType t) { return BytesOf(t); });

    m.def("begin_function", [](const std::string &funcName, GraphType graphType, FunctionType funcType, py::args args) {
        std::vector<std::reference_wrapper<Tensor>> tensors;
        tensors.reserve(args.size());
        for (auto &a : args) {
            tensors.push_back(a.cast<Tensor &>());
        }
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        Program::GetInstance().BeginFunction(FUNCTION_PREFIX + funcName, funcType, graphType, tensors);
    });

    m.def("end_function", [](const std::string &funcName, bool generateCall) {
        Program::GetInstance().EndFunction(FUNCTION_PREFIX + funcName, generateCall);
    });

    m.def("dump", []() { return Program::GetInstance().Dump(); });

    py::class_<RecordFunc>(m, "record_func")
        .def(py::init<const std::string &>(), py::arg("name"))
        .def(py::init<const std::string &, const FunctionConfig>(), py::arg("name"), py::arg("func_config"))
        .def(py::init<const std::string &, const FunctionConfig, const std::vector<std::reference_wrapper<Tensor>> &>(),
            py::arg("name"), py::arg("func_config"), py::arg("explicit_op_args"))
        .def(
            py::init<const std::string &, const FunctionConfig&, const ref_tensors &, const ref_tensors &,
                const std::vector<std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>>
                    &>(),
            py::arg("name"), py::arg("func_config"), py::arg("start_args_input_tensor_list"),
            py::arg("start_args_output_tensor_list"), py::arg("in_place_args"));

    py::class_<RecordLoopFunc>(m, "record_loop_func")
        .def(py::init<const std::string &, FunctionType, const std::string &, const LoopRange &, const std::set<int> &,
                 bool>(),
            py::arg("name"), py::arg("func_type"), py::arg("iter_name"), py::arg("loop_range"), py::arg("unroll_List"),
            py::arg("submit_before_loop"))
        .def("begin_loop_func", &RecordLoopFunc::BeginLoopFunction)
        .def("end_loop_func", &RecordLoopFunc::EndLoopFunction)
        .def("iteration_begin", &RecordLoopFunc::IterationBegin)
        .def("iteration_next", &RecordLoopFunc::IterationNext)
        .def("iteration_end", &RecordLoopFunc::IterationEnd)
        .def(
            "__iter__",
            [](RecordLoopFunc &c) {
                // Return Python iterator from C++ begin/end
                return py::make_iterator(c.begin(), c.end());
            },
            py::keep_alive<0, 1>()); // Keep container alive while iterator is used;

    py::class_<RecordLoopFunc::Iterator>(m, "record_loop_func_iter")
        .def(py::init<RecordLoopFunc &, const SymbolicScalar &>(), py::arg("rlf"), py::arg("scalar"));

    py::class_<RecordLoopFunc::IteratorEnd>(m, "record_loop_func_iter_end")
        .def(py::init<RecordLoopFunc &, const SymbolicScalar &>(), py::arg("rlf"), py::arg("scalar"));

    py::class_<RecordIfBranch>(m, "record_if_branch")
        .def(py::init<SymbolicScalar, const std::string &, int>(), py::arg("cond"), py::arg("file") = "",
            py::arg("line") = 0)
        .def("__bool__", py::overload_cast<>(&RecordIfBranch::operator bool, py::const_));

    // (anastasios): not used now since we use Program::GetInstance()
    py::class_<TileShape>(m, "tile_shape")
        .def(py::init<>())
        .def("reset", &TileShape::Reset)
        .def("to_string", &TileShape::toString, py::arg("tile_type") = TileType::MAX)
        .def("get_vec_tile_shapes", py::overload_cast<>(&TileShape::GetVecTile))
        .def("get_vec_tile_shapes", py::overload_cast<>(&TileShape::GetVecTile, py::const_))
        .def("set_vec_tile_shapes", [](TileShape &self, py::args args) {
            std::vector<int64_t> v;
            v.reserve(args.size());
            for (auto &a : args) {
                v.push_back(a.cast<int>()); // require ints
            }
            self.SetVecTile(v);
        })
        .def("set_disk_rank_id", &TileShape::SetDistRankId);

    py::class_<VecTile>(m, "vec_tile")
        .def(py::init<>())
        .def_readwrite("tile", &VecTile::tile)
        .def("valid", &VecTile::valid,
             "Check if all elements are positive and non-empty")
        .def("__getitem__", [](const VecTile& vt, int index) {
            if (index < 0 || index >= static_cast<int>(vt.size())) {
                throw py::index_error("Index out of range");
            }
            return vt[index];
        }, py::arg("index"))
        .def("__len__", &VecTile::size, "Get the size of the tile");

    py::class_<LoopRange>(m, "loop_range_")
        .def(py::init<const SymbolicScalar & /* rangeBegin */, const SymbolicScalar & /* rangeEnd */,
            const SymbolicScalar & /* rangeStep */>())
        .def(py::init<const SymbolicScalar & /* rangeBegin */, const SymbolicScalar & /* rangeEnd */>())
        .def(py::init<const SymbolicScalar & /* rangeEnd */>())
        .def(py::init<std::int64_t>()) // C++ Implicit conversion int64_t -> SymbolicScalar
        .def("dump", (std::string (LoopRange::*)())&LoopRange::Dump)
        .def("begin", (SymbolicScalar& (LoopRange::*)()) & LoopRange::Begin,
            py::return_value_policy::reference_internal)
        .def("end", (SymbolicScalar& (LoopRange::*)()) & LoopRange::End, py::return_value_policy::reference_internal)  
        .def(
            "step", (SymbolicScalar& (LoopRange::*)()) & LoopRange::Step, py::return_value_policy::reference_internal); 

    py::class_<FunctionConfig>(m, "func_config")
        .def(py::init<>())
        .def(py::init<FunctionType>(),
             py::arg("funcType") = FunctionType::DYNAMIC)
        .def_readwrite("funcType", &FunctionConfig::funcType);

    m.def("is_loop_begin_", &IsLoopBegin, py::arg("symbol"), py::arg("begin"));
    m.def("is_loop_end_", &IsLoopEnd, py::arg("symbol"), py::arg("end"));
    m.def("powers_of_2", &PowersOf2, py::arg("n"));
}
} // namespace pypto
