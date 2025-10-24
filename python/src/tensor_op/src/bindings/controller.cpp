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

enum class OptionType {
    String,    // std::string
    Int,       // int
    VectorInt, // std::vector<int>
    MapIntInt  // std::map<int, int>
};

static std::map<std::string, OptionType> PassKeyTypeMap;
static std::map<std::string, OptionType> HostKeyTypeMap;
static std::map<std::string, OptionType> CodeGenKeyTypeMap;
static std::map<std::string, OptionType> RuntimeTypeMap;

namespace pypto {
void bind_controller_config(py::module &m) {
    m.def("SetBuildStatic", [](const bool &value) { config::SetBuildStatic(value); }, py::arg("value"));
    m.def(
        "SetOption", [](const std::string &key, const std::string &value) { config::SetOption(key, value); },
        py::arg("key"), py::arg("value"));
    m.def(
        "SetOption", [](const std::string &key, bool value) { config::SetOption(key, value); }, py::arg("key"),
        py::arg("value"));
    m.def(
        "SetOption", [](const std::string &key, const std::vector<int64_t> &value) { config::SetOption(key, value); },
        py::arg("key"), py::arg("value"));
    m.def(
        "SetOption",
        [](const std::string &key, const std::map<int64_t, int64_t> &value) { config::SetOption(key, value); },
        py::arg("key"), py::arg("value"));

    m.def(
        "SetPrintOptions",
        [](int edgeItems, int precision, int threshold, int linewidth) { config::SetPrintOptions(edgeItems, precision, threshold, linewidth); },
        py::arg("edgeItems"), py::arg("precision"), py::arg("threshold"), py::arg("linewidth"));
    m.def(
        "GetPrintOptions", 
        &config::GetPrintOptions,
        py::return_value_policy::reference,
        "Get reference to the global print options configuration"
    );
}

void bind_print_options(py::module &m) {
    py::class_<PrintOptions>(m, "print_options")
        .def(py::init<>())
        .def_readwrite("edge_items", &PrintOptions::edgeItems)
        .def_readwrite("precision", &PrintOptions::precision)
        .def_readwrite("threshold", &PrintOptions::threshold)
        .def_readwrite("linewidth", &PrintOptions::linewidth);
}

void bind_config_host_option(py::module &m){
        m.def(
        "SetHostOption",
        [](const std::string &key, const py::object &value) {
            if (py::isinstance<py::int_>(value)) {
                config::SetHostOption(key, value.cast<int64_t>());
                HostKeyTypeMap[key] = OptionType::Int;
            } else if (py::isinstance<py::str>(value)) {
                config::SetHostOption(key, value.cast<std::string>());
                HostKeyTypeMap[key] = OptionType::String;
            } else if (py::isinstance<py::list>(value)) {
                config::SetHostOption(key, value.cast<std::vector<int64_t>>());
                HostKeyTypeMap[key] = OptionType::VectorInt;
            } else if (py::isinstance<py::dict>(value)) {
                config::SetHostOption(key, value.cast<std::map<int64_t, int64_t>>());
                HostKeyTypeMap[key] = OptionType::MapIntInt;
            } else {
                throw std::runtime_error(
                    "Unsupported value type for SetHostOption. Expected one of: str, int, List[int], Dict[int, int]");
            }
        },
        py::arg("key"), py::arg("value"),
        "Set a host option (accepts str, int, List[int], or Dict[int, int]).");

    m.def(
        "GetHostOption",
        [](const std::string& key) -> py::object {
            auto type_iter = HostKeyTypeMap.find(key);
            if (type_iter == HostKeyTypeMap.end()) {
                throw std::invalid_argument("Key not found: " + key + " (not set via SetHostOption)");
            }
            OptionType key_type = type_iter->second;
            switch (key_type) {
                case OptionType::String: {
                    return py::cast(config::GetHostOption<std::string>(key));
                }
                case OptionType::Int: {
                    return py::cast(config::GetHostOption<int64_t>(key));
                }
                case OptionType::VectorInt: {
                    return py::cast(config::GetHostOption<std::vector<int64_t>>(key));
                }
                case OptionType::MapIntInt: {
                    return py::cast(config::GetHostOption<std::map<int64_t, int64_t>>(key));
                }
                default:
                    throw std::runtime_error("Unknown option type");
            }
        },
        py::arg("key"),
        "Get a host configuration item (returns the same type as set via SetHostOption).");
}

void bind_config_pass_option(py::module &m){
    m.def(
        "SetPassOption",
        [](const std::string &key, const py::object &value) {
            if (py::isinstance<py::int_>(value)) {
                config::SetPassOption(key, value.cast<int64_t>());
                PassKeyTypeMap[key] = OptionType::Int;
            }else if (py::isinstance<py::str>(value)) {
                config::SetPassOption(key, value.cast<std::string>());
                PassKeyTypeMap[key] = OptionType::String;
            }else if (py::isinstance<py::list>(value)) {
                config::SetPassOption(key, value.cast<std::vector<int64_t>>());
                PassKeyTypeMap[key] = OptionType::VectorInt;
            } else if (py::isinstance<py::dict>(value)) {
                config::SetPassOption(key, value.cast<std::map<int64_t, int64_t>>());
                PassKeyTypeMap[key] = OptionType::MapIntInt;
            } else {
                throw std::runtime_error("Unsupported value type for set_pass_option. Expected Union[str, int, List[int], Dict[int, int]]");
            }
        },
        py::arg("key"), py::arg("value"),
        "Set pass option with key-value pair. Value type: Union[str, int, List[int], Dict[int, int]].");

    m.def(
        "GetPassOption",
        [](const std::string& key) -> py::object {
            auto type_iter = PassKeyTypeMap.find(key);
            if (type_iter == PassKeyTypeMap.end()) {
                throw std::invalid_argument("Key not found: " + key + " (not set via SetPassOption)");
            }
            OptionType key_type = type_iter->second;
            switch (key_type) {
                case OptionType::String: {
                    return py::cast(config::GetPassOption<std::string>(key));
                }
                case OptionType::Int: {
                    return py::cast(config::GetPassOption<int64_t>(key));
                }
                case OptionType::VectorInt: {
                    return py::cast(config::GetPassOption<std::vector<int64_t>>(key));
                }
                case OptionType::MapIntInt: {
                    return py::cast(config::GetPassOption<std::map<int64_t, int64_t>>(key));
                }
                default:
                    throw std::runtime_error("Unknown option type");
            }
        },
        py::arg("key"),
        "Get a configuration item (returns the same type as set via SetPassOption).");

    m.def(
        "SetCodeGenOption",        
        [](const std::string &key, const std::string &value) {
            config::SetCodeGenOption<std::string>(key, value);
            CodeGenKeyTypeMap[key] = OptionType::String;
        },
        py::arg("key"), py::arg("value")
    );
    m.def(
        "SetCodeGenOption",
        [](const std::string &key, const int64_t &value) {
            config::SetCodeGenOption<int64_t>(key, value);
            CodeGenKeyTypeMap[key] = OptionType::Int;
        },
        py::arg("key"), py::arg("value")
    );
    m.def(
        "SetCodeGenOption",
        [](const std::string& key, const std::vector<int64_t>& value) {
            config::SetCodeGenOption<std::vector<int64_t>>(key, value);
            CodeGenKeyTypeMap[key] = OptionType::VectorInt;
        },
        py::arg("key"), py::arg("value")
    );
    m.def(
        "SetCodeGenOption",
        [](const std::string &key, const std::map<int64_t, int64_t> &value) {
            config::SetCodeGenOption<std::map<int64_t, int64_t>>(key, value);
            CodeGenKeyTypeMap[key] = OptionType::MapIntInt;
        },
        py::arg("key"), py::arg("value")
    );

    m.def(
        "GetCodeGenOption",
        [](const std::string& key) -> py::object {
            auto type_iter = CodeGenKeyTypeMap.find(key);
            if (type_iter == CodeGenKeyTypeMap.end()) {
                throw std::invalid_argument("Key not found: " + key + " (Not set via SetCodeGenOption)");
            }
            OptionType key_type = type_iter->second;

            switch (key_type) {
                case OptionType::String: {
                    return py::cast(config::GetCodeGenOption<std::string>(key));
                }
                case OptionType::Int: {
                    return py::cast(config::GetCodeGenOption<int64_t>(key));
                }
                case OptionType::VectorInt: {
                    return py::cast(config::GetCodeGenOption<std::vector<int64_t>>(key));
                }
                case OptionType::MapIntInt: {
                    return py::cast(config::GetCodeGenOption<std::map<int64_t, int64_t>>(key));
                }
                default:
                    throw std::invalid_argument("Unknown type，key: " + key);
            }
        },
        py::arg("key"),
        "Get the configuration item (the return type is consistent with when it was set via SetCodeGenOption)"
    );
     
    m.def(
        "SetRuntimeOption",
        [](const std::string &key, const std::string &value) {
            config::SetRuntimeOption<std::string>(key, value);
            RuntimeTypeMap[key] = OptionType::String;
        },
        py::arg("key"), py::arg("value")
    );
    m.def(
        "SetRuntimeOption",
        [](const std::string &key, const int64_t &value) {
            config::SetRuntimeOption<int64_t>(key, value);
            RuntimeTypeMap[key] = OptionType::Int;
        },
        py::arg("key"), py::arg("value")
    );
    m.def(
        "SetRuntimeOption",
        [](const std::string& key, const std::vector<int64_t>& value) {
            config::SetRuntimeOption<std::vector<int64_t>>(key, value);
            RuntimeTypeMap[key] = OptionType::VectorInt;
        },
        py::arg("key"), py::arg("value")
    );
    m.def(
        "SetRuntimeOption",
        [](const std::string &key, const std::map<int64_t, int64_t> &value) {
            config::SetRuntimeOption<std::map<int64_t, int64_t>>(key, value);
            RuntimeTypeMap[key] = OptionType::MapIntInt;
        },
        py::arg("key"), py::arg("value")
    );

    m.def(
        "GetRuntimeOption",
        [](const std::string& key) -> py::object {
            auto type_iter = RuntimeTypeMap.find(key);
            if (type_iter == RuntimeTypeMap.end()) {
                throw std::invalid_argument("Key not found: " + key + "(Not set via SetCodeGenOption)");
            }
            OptionType key_type = type_iter->second;

            switch (key_type) {
                case OptionType::String: {
                    return py::cast(config::GetRuntimeOption<std::string>(key));
                }
                case OptionType::Int: {
                    return py::cast(config::GetRuntimeOption<int64_t>(key));
                }
                case OptionType::VectorInt: {
                    return py::cast(config::GetRuntimeOption<std::vector<int64_t>>(key));
                }
                case OptionType::MapIntInt: {
                    return py::cast(config::GetRuntimeOption<std::map<int64_t, int64_t>>(key));
                }
                default:
                    throw std::invalid_argument("Unknown type，key: " + key);
            }
        },
        py::arg("key"),
        "Get the configuration item (the return type is consistent with when it was set via SetRuntimeOption)"
    );

    m.def(
        "SetSemanticLabel",
        [](const std::string &label) { config::SetSemanticLabel(label); },
        py::arg("label"));    
}

void bind_controller_tile_shape(py::module &m) {
    py::class_<TileShape>(m, "TileShape")
        .def(py::init<>())
        .def("Reset", &TileShape::Reset)
        .def("toString", &TileShape::toString, py::arg("tile_type") = TileType::MAX)
        .def("GetVecTile", py::overload_cast<>(&TileShape::GetVecTile))
        .def("GetVecTile", py::overload_cast<>(&TileShape::GetVecTile, py::const_))
        .def("GetCubeTile", py::overload_cast<>(&TileShape::GetCubeTile))
        .def("GetCubeTile", py::overload_cast<>(&TileShape::GetCubeTile, py::const_))
        .def("SetVecTile",
            [](TileShape &self, py::args args) {
                std::vector<int64_t> v;
                v.reserve(args.size());
                for (auto &a : args) {
                    v.push_back(a.cast<int>()); // require ints
                }
                self.SetVecTile(v);
            })
        .def("SetDistRankId", &TileShape::SetDistRankId);
    py::class_<VecTile>(m, "VecTile")
        .def(py::init<>())
        .def_readwrite("tile", &VecTile::tile)
        .def("valid", &VecTile::valid, "Check if all elements are positive and non-empty")
        .def(
            "__getitem__",
            [](const VecTile &vt, int index) {
                if (index < 0 || index >= static_cast<int>(vt.size())) {
                    throw py::index_error("Index out of range");
                }
                return vt[index];
            },
            py::arg("index"))
        .def("__len__", &VecTile::size, "Get the size of the tile");
}

void bind_controller_set_tile(py::module &m) {
    m.def("SetVecTile", [](py::args args) {
        std::vector<int64_t> v;
        v.reserve(args.size());
        for (auto &a : args) {
            v.push_back(a.cast<int64_t>());
        }
        TileShape::Current().SetVecTile(v);
    });
    m.def("GetVecTile", []() { return TileShape::Current().GetVecTile(); });
    m.def(
        "SetMatrixSize", [](const std::vector<int64_t> &size) { TileShape::Current().SetMatrixSize(size); },
        py::arg("size"));
    m.def(
        "SetCubeTile",
        [](const std::vector<int64_t> &mvec, const std::vector<int64_t> &kvec, const std::vector<int64_t> &nvec,
            bool setL1Tile) {
            if (mvec.size() > MAX_M_DIM_SIZE) {
                throw py::value_error(
                    "Parameter 'm' must have exactly " + std::to_string(MAX_M_DIM_SIZE) + " elements");
            }
            if (kvec.size() > MAX_K_DIM_SIZE) {
                throw py::value_error(
                    "Parameter 'k' must have exactly " + std::to_string(MAX_K_DIM_SIZE) + " elements");
            }
            if (nvec.size() > MAX_N_DIM_SIZE) {
                throw py::value_error(
                    "Parameter 'n' must have exactly " + std::to_string(MAX_N_DIM_SIZE) + " elements");
            }

            std::array<int64_t, MAX_M_DIM_SIZE> marr = {0};
            std::array<int64_t, MAX_K_DIM_SIZE> karr = {0};
            std::array<int64_t, MAX_N_DIM_SIZE> narr = {0};

            std::copy(mvec.begin(), mvec.end(), marr.begin());
            std::copy(kvec.begin(), kvec.end(), karr.begin());
            std::copy(nvec.begin(), nvec.end(), narr.begin());
            TileShape::Current().SetCubeTile(marr, karr, narr, setL1Tile);
        },
        py::arg("m"), py::arg("k"), py::arg("n"), py::arg("set_l1_tile"),
        "Set cube tile shapes with specified dimensions");
    m.def("GetCubeTile", []() { return TileShape::Current().GetCubeTile(); });
}

void bind_controller_function(py::module &m) {
    m.def("BeginFunction", [](const std::string &funcName, GraphType graphType, FunctionType funcType, py::args args) {
        std::vector<std::reference_wrapper<Tensor>> tensors;
        tensors.reserve(args.size());
        for (auto &a : args) {
            tensors.push_back(a.cast<Tensor &>());
        }
        Program::GetInstance().Reset();
        config::Reset();
        Program::GetInstance().BeginFunction(FUNCTION_PREFIX + funcName, funcType, graphType, tensors);
    });
    m.def("EndFunction", [](const std::string &funcName, bool generateCall) {
        Program::GetInstance().EndFunction(FUNCTION_PREFIX + funcName, generateCall);
    });
    py::class_<RecordFunc>(m, "RecordFunc")
        .def(py::init<const std::string &>(), py::arg("name"))
        .def(py::init<const std::string &, const std::vector<std::reference_wrapper<Tensor>> &>(), py::arg("name"),
            py::arg("explicit_op_args"))
        .def(
            py::init<const std::string &, const ref_tensors &, const ref_tensors &,
                const std::vector<std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>>
                    &>(),
            py::arg("name"), py::arg("start_args_input_tensor_list"), py::arg("start_args_output_tensor_list"),
            py::arg("in_place_args"));
    py::class_<RecordLoopFunc>(m, "RecordLoopFunc")
        .def(py::init<const std::string &, FunctionType, const std::string &, const LoopRange &, const std::set<int> &,
                 bool>(),
            py::arg("name"), py::arg("func_type"), py::arg("iter_name"), py::arg("loop_range"), py::arg("unroll_List"),
            py::arg("submit_before_loop"))
        .def("BeginLoopFunction", &RecordLoopFunc::BeginLoopFunction)
        .def("EndLoopFunction", &RecordLoopFunc::EndLoopFunction)
        .def("IterationBegin", &RecordLoopFunc::IterationBegin)
        .def("IterationNext", &RecordLoopFunc::IterationNext)
        .def("IterationEnd", &RecordLoopFunc::IterationEnd)
        .def("MatchUnrollTimes", &RecordLoopFunc::MatchUnrollTimes)
        .def("__iter__", [](RecordLoopFunc &c) {
            // Return Python iterator from C++ begin/end
            return py::make_iterator(c.begin(), c.end());
        });

    py::class_<RecordLoopFunc::Iterator>(m, "RecordLoopFunc_Iterator")
        .def(py::init<RecordLoopFunc &, const SymbolicScalar &>(), py::arg("rlf"), py::arg("scalar"));
    py::class_<RecordLoopFunc::IteratorEnd>(m, "RecordLoopFunc_IteratorEnd")
        .def(py::init<RecordLoopFunc &, const SymbolicScalar &>(), py::arg("rlf"), py::arg("scalar"));
}

void bind_controller_loop(py::module &m) {
    py::class_<RecordIfBranch>(m, "RecordIfBranch")
        .def(py::init<SymbolicScalar, const std::string &, int>(), py::arg("cond"), py::arg("file") = "",
            py::arg("line") = 0)
        .def("__bool__", py::overload_cast<>(&RecordIfBranch::operator bool, py::const_));
    py::class_<LoopRange>(m, "LoopRange")
        .def(py::init<const SymbolicScalar & /* rangeBegin */, const SymbolicScalar & /* rangeEnd */,
            const SymbolicScalar & /* rangeStep */>())
        .def(py::init<const SymbolicScalar & /* rangeBegin */, const SymbolicScalar & /* rangeEnd */>())
        .def(py::init<const SymbolicScalar & /* rangeEnd */>())
        .def(py::init<std::int64_t>()) // C++ Implicit conversion int64_t -> SymbolicScalar
        .def("Dump", (std::string(LoopRange::*)()) & LoopRange::Dump)
        .def("Begin", (SymbolicScalar & (LoopRange::*)()) & LoopRange::Begin,
            py::return_value_policy::reference_internal)
        .def("End", (SymbolicScalar & (LoopRange::*)()) & LoopRange::End, py::return_value_policy::reference_internal)
        .def(
            "Step", (SymbolicScalar & (LoopRange::*)()) & LoopRange::Step, py::return_value_policy::reference_internal);

    m.def("IsLoopBegin", &IsLoopBegin, py::arg("symbol"), py::arg("begin"));
    m.def("IsLoopEnd", &IsLoopEnd, py::arg("symbol"), py::arg("end"));
}

void bind_controller_utils(py::module &m) {
    m.def("Dump", []() { return Program::GetInstance().Dump(); });
    m.def("SetSemanticLabel", [](const std::string &label) { config::SetSemanticLabel(label); }, py::arg("label"));
    m.def("BytesOf", [](DataType t) { return BytesOf(t); });
    m.def("Reset", []() { Program::GetInstance().Reset(); });
}

void bind_controller(py::module &m) {
    bind_print_options(m);  
    bind_config_pass_option(m);
    bind_config_host_option(m);
    bind_controller_config(m);
    bind_controller_tile_shape(m);
    bind_controller_set_tile(m);
    bind_controller_function(m);
    bind_controller_loop(m);
    bind_controller_utils(m);
}
} // namespace pypto
