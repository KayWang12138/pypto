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
 * \file controller.cpp
 * \brief
 */

#include "nb_common.h"
#include "tilefwk/function.h"
#include "tilefwk/tensor.h"

#include <functional>
#include <utility>
#include <vector>
#include <string>
#include <typeindex>

using namespace npu::tile_fwk;
using ref_tensors = std::vector<std::reference_wrapper<const Tensor>>;

namespace pypto {
void bind_controller_config(nb::module_ &m) {
    m.def("SetBuildStatic", [](const bool &value) { config::SetBuildStatic(value); }, nb::arg("value"));

    m.def("ResetOptions", []() {
        config::Reset();
    });


    m.def(
        "SetPrintOptions",
        [](int edgeItems, int precision, int threshold, int linewidth) {
            config::SetPrintOptions(edgeItems, precision, threshold, linewidth);
        },
        nb::arg("edgeItems"), nb::arg("precision"), nb::arg("threshold"), nb::arg("linewidth"));

    m.def("SetSemanticLabel",
        [](const std::string &label, const std::string &filename, int lineno) {
            config::SetSemanticLabel(label, filename.c_str(), lineno);
        }, nb::arg("label"), nb::arg("filename"), nb::arg("lineno"));

    m.def("IsVerifyEnabled", &calc::IsVerifyEnabled);
    m.def("LogTopFolder", []() { return nb::cast(ConfigManager::Instance().LogTopFolder()); });
    m.def("ResetLog", [](const std::string &path) { ConfigManager::Instance().ResetLog(path); });
}


void bind_controller_set_tile(nb::module_ &m) {
    m.def("SetVecTile", [](nb::args &args) {
        std::vector<int64_t> v;
        v.reserve(args.size());
        for (auto a : args) {
            v.push_back(nb::cast<int64_t>(a));
        }
        TileShape::Current().SetVecTile(v);
    });
    m.def("GetVecTile", []() { return TileShape::Current().GetVecTile().tile; });
    m.def(
        "SetMatrixSize", [](const std::vector<int64_t> &size) { TileShape::Current().SetMatrixSize(size); },
        nb::arg("size"));
    m.def(
        "SetCubeTile",
        [](const std::vector<int64_t> &mvec, const std::vector<int64_t> &kvec, const std::vector<int64_t> &nvec,
            bool enableMultiDataLoad, bool enableSplitK) {
            if (mvec.size() > MAX_M_DIM_SIZE) {
                throw nb::value_error("Parameter 'm' must have exactly 2 elements");
            }
            if (kvec.size() > MAX_K_DIM_SIZE) {
                throw nb::value_error("Parameter 'k' must have exactly 3 elements");
            }
            if (nvec.size() > MAX_N_DIM_SIZE) {
                throw nb::value_error("Parameter 'n' must have exactly 2 elements");
            }

            std::array<int64_t, MAX_M_DIM_SIZE> marr = {0};
            std::array<int64_t, MAX_K_DIM_SIZE> karr = {0};
            std::array<int64_t, MAX_N_DIM_SIZE> narr = {0};

            std::copy(mvec.begin(), mvec.end(), marr.begin());
            std::copy(kvec.begin(), kvec.end(), karr.begin());
            std::copy(nvec.begin(), nvec.end(), narr.begin());
            TileShape::Current().SetCubeTile(marr, karr, narr, enableMultiDataLoad, enableSplitK);
        },
        nb::arg("m"), nb::arg("k"), nb::arg("n"), nb::arg("enable_multi_data_load"), nb::arg("enable_split_k"),
        "Set cube tile shapes with specified dimensions");
    m.def("GetCubeTile", []() {
        auto cubeTile = TileShape::Current().GetCubeTile();
        return std::tuple(cubeTile.m, cubeTile.k, cubeTile.n, cubeTile.enableMultiDataLoad, cubeTile.enableSplitK);
    });
}

void bind_controller_function(nb::module_ &m) {
    m.def("BeginFunction", [](const std::string &funcName, GraphType graphType, FunctionType funcType, nb::args &args) {
        std::vector<std::reference_wrapper<const Tensor>> tensors;
        tensors.reserve(args.size());
        for (auto a : args) {
            tensors.push_back(nb::cast<Tensor &>(a));
        }
        Program::GetInstance().Reset();
        Program::GetInstance().BeginFunction(FUNCTION_PREFIX + funcName, funcType, graphType, tensors);
    });
    m.def("EndFunction", [](const std::string &funcName, bool generateCall) {
        Program::GetInstance().EndFunction(FUNCTION_PREFIX + funcName, generateCall);
    });
    nb::class_<RecordFunc>(m, "RecordFunc")
        .def(nb::init<const std::string &>(), nb::arg("name"))
        .def("__init__", [](RecordFunc *self, const std::string &name, nb::list &args) {
            std::vector<std::reference_wrapper<const Tensor>> tensors;
            for (auto a : args) {
                tensors.push_back(nb::cast<const Tensor &>(a));
            }
            new (self) RecordFunc(name, tensors);
        }, nb::arg("name"), nb::arg("args"))
        .def("EndFunction", &RecordFunc::EndFunction)
        .def("__iter__", [](RecordFunc &c) {
            // Return Python iterator from C++ begin/end
            return nb::make_iterator(nb::type<RecordFunc>(), "Iterator", c.begin(), c.end());
            }, nb::rv_policy::reference_internal);
    nb::class_<RecordLoopFunc>(m, "RecordLoopFunc")
        .def(nb::init<const std::string &, FunctionType, const std::string &, const LoopRange &, const std::set<int> &, bool>(),
            nb::arg("name"), nb::arg("func_type"), nb::arg("iter_name"), nb::arg("loop_range"), nb::arg("unroll_List"),
            nb::arg("submit_before_loop"))
        .def("__iter__", [](RecordLoopFunc &c) {
            // Return Python iterator from C++ begin/end
            return nb::make_iterator(nb::type<RecordLoopFunc>(), "Iterator", c.begin(), c.end());
        });
}

void bind_controller_loop(nb::module_ &m) {
    nb::class_<RecordIfBranch>(m, "RecordIfBranch")
        .def(nb::init<SymbolicScalar, const std::string &, int>(), nb::arg("cond"), nb::arg("file") = "",
            nb::arg("line") = 0)
        .def("__bool__", nb::overload_cast<>(&RecordIfBranch::operator bool, nb::const_));
    nb::class_<LoopRange>(m, "LoopRange")
        .def(nb::init<const SymbolicScalar &/* rangeBegin */, const SymbolicScalar &/* rangeEnd */,
            const SymbolicScalar &/* rangeStep */>())
        .def(nb::init<const SymbolicScalar &/* rangeBegin */, const SymbolicScalar &/* rangeEnd */>())
        .def(nb::init<const SymbolicScalar &/* rangeEnd */>())
        .def(nb::init<std::int64_t>()) // C++ Implicit conversion int64_t -> SymbolicScalar
        .def("Dump", (std::string(LoopRange::*)()) &LoopRange::Dump)
        .def("Begin", (SymbolicScalar &(LoopRange::*)()) &LoopRange::Begin,
            nb::rv_policy::reference_internal)
        .def("End", (SymbolicScalar &(LoopRange::*)()) &LoopRange::End, nb::rv_policy::reference_internal)
        .def(
            "Step", (SymbolicScalar &(LoopRange::*)()) &LoopRange::Step, nb::rv_policy::reference_internal);

    m.def("IsLoopBegin", &IsLoopBegin, nb::arg("symbol"), nb::arg("begin"));
    m.def("IsLoopEnd", &IsLoopEnd, nb::arg("symbol"), nb::arg("end"));
}

void bind_controller_utils(nb::module_ &m) {
    m.def("Dump", []() { return Program::GetInstance().Dump(); });
    m.def("BytesOf", [](DataType t) { return BytesOf(t); });
    m.def("Reset", []() { Program::GetInstance().Reset(); });
    m.def("SetLocation", [](const std::string &fname, int lineno) {
        SourceLocation::SetLocation(fname, lineno);
    }, nb::arg("fname"), nb::arg("lineno"));
    m.def("SetLocation", [](const std::string &fname, int lineno, std::string &backtrace) {
        SourceLocation::SetLocation(fname, lineno, backtrace);
    }, nb::arg("fname"), nb::arg("lineno"), nb::arg("backtrace"));
    m.def("ClearLocation", &SourceLocation::ClearLocation);
}


std::map<std::string, npu::tile_fwk::Any> ConvertPyDictToCppMap(const nb::dict &values) {
    std::map<std::string, npu::tile_fwk::Any> cpp_values;
    for (auto item : values) {
        std::string key = nb::cast<std::string>(item.first);
        auto &value = item.second;

        if (nb::isinstance<nb::bool_>(value)) {
            cpp_values[key] = nb::cast<bool>(value);
        } else if (nb::isinstance<nb::int_>(value)) {
            cpp_values[key] = nb::cast<int64_t>(value);
        } else if (nb::isinstance<nb::float_>(value)) {
            cpp_values[key] = nb::cast<double>(value);
        } else if (nb::isinstance<nb::str>(value)) {
            cpp_values[key] = nb::cast<std::string>(value);
        } else if (nb::isinstance<CubeTile>(value)) {
            cpp_values[key] = nb::cast<CubeTile>(value);
        } else if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
            nb::list lst = nb::cast<nb::list>(value);
            if (lst.size() > 0) {
                if (nb::isinstance<nb::int_>(lst[0])) {
                    cpp_values[key] = nb::cast<std::vector<int64_t>>(lst);
                } else if (nb::isinstance<nb::str>(lst[0])) {
                    cpp_values[key] = nb::cast<std::vector<std::string>>(lst);
                } else if (nb::isinstance<nb::float_>(lst[0])) {
                    cpp_values[key] = nb::cast<std::vector<double>>(lst);
                } else {
                    throw nb::type_error(("Unsupported list element type for key: " + key).c_str());
                }
            } else {
                cpp_values[key] = std::vector<int64_t>();
            }
        } else if (nb::isinstance<nb::dict>(value)) {
            cpp_values[key] = nb::cast<std::map<int64_t, int64_t>>(value);
        } else {
            throw nb::type_error(("Unsupported value type for key: " + key).c_str());
        }
    }

    return cpp_values;
}

void bind_controller_scope(nb::module_ &m) {
    m.def("BeginScope",
        [](const std::string &name, const nb::dict &values,
        const std::string &filename, int lineno) {
            auto cpp_values = ConvertPyDictToCppMap(values);
            ConfigManagerNg::GetInstance().BeginScope(name, std::move(cpp_values), filename.c_str(), lineno);
        },
        nb::arg("name"),
        nb::arg("values"),
        nb::arg("filename"),
        nb::arg("lineno")
    );

    m.def("EndScope",
        [](const std::string &filename, int lineno) {
            ConfigManagerNg::GetInstance().EndScope(filename.c_str(), lineno);
        },
        nb::arg("filename") = "default",
        nb::arg("lineno") = -1
    );

    m.def("SetScope",
        [](const nb::dict &values, const std::string &filename, int lineno) {
            auto cpp_values = ConvertPyDictToCppMap(values);
            ConfigManagerNg::GetInstance().SetScope(std::move(cpp_values), filename.c_str(), lineno);
        },
        nb::arg("values"),
        nb::arg("filename") = "default",
        nb::arg("lineno") = -1
    );

    m.def("SetGlobalConfig",
        [](const nb::dict &values, const std::string &filename, int lineno) {
            auto cpp_values = ConvertPyDictToCppMap(values);
            ConfigManagerNg::GetInstance().SetGlobalConfig(std::move(cpp_values), filename.c_str(), lineno);
        },
        nb::arg("values"),
        nb::arg("filename") = "default",
        nb::arg("lineno") = -1
    );

    m.def("CurrentScope",
        []() { return ConfigManagerNg::GetInstance().CurrentScope(); });

    m.def("GlobalScope",
        []() { return ConfigManagerNg::GetInstance().GlobalScope(); });

    m.def("GetOptionsTree",
        []() { return ConfigManagerNg::GetInstance().GetOptionsTree(); });
}

nb::object AnyToPyObject(const Any &val) {
    using Fn = std::function<nb::object(const Any&)>;
    static const std::unordered_map<std::type_index, Fn> table = {
        {typeid(bool), [](const Any& a){ return nb::cast(AnyCast<bool>(a)); }},
        {typeid(int64_t), [](const Any& a){ return nb::cast(AnyCast<int64_t>(a)); }},
        {typeid(double), [](const Any& a){ return nb::cast(AnyCast<double>(a)); }},
        {typeid(std::string), [](const Any& a){ return nb::cast(AnyCast<std::string>(a)); }},
        {typeid(std::vector<int64_t>), [](const Any& a){ return nb::cast(AnyCast<std::vector<int64_t>>(a)); }},
        {typeid(std::vector<std::string>), [](const Any& a){ return nb::cast(AnyCast<std::vector<std::string>>(a)); }},
        {typeid(std::vector<double>), [](const Any& a){ return nb::cast(AnyCast<std::vector<double>>(a)); }},
        {typeid(std::map<int64_t,int64_t>), [](const Any& a){ return nb::cast(AnyCast<std::map<int64_t,int64_t>>(a)); }},
        {typeid(CubeTile), [](const Any& a){ return nb::cast(AnyCast<CubeTile>(a)); }},
        {typeid(DistTile), [](const Any& a){ return nb::cast(AnyCast<DistTile>(a).ToString()); }},
    };

    auto it = table.find(std::type_index(val.Type()));
    if (it != table.end()) return it->second(val);

    throw nb::type_error(("Unsupported config value type: " + std::string(val.Type().name())).c_str());
}

void bind_controller_scope_classes(nb::module_ &m) {
    nb::class_<ConfigScope>(m, "ConfigScope")
        .def("GetAnyConfig",
            [](const ConfigScope &scope, const std::string &key) -> nb::object {
                return AnyToPyObject(scope.GetAnyConfig(key));
            },
            nb::arg("key"))
        .def("GetAllConfig",
            [](const ConfigScope &scope) -> nb::dict {
                nb::dict result;
                auto config_map = scope.GetAllConfig();

                for (const auto &[key, val] : config_map) {
                        result[nb::cast(key)] = AnyToPyObject(val);
                }
                return result;
            })
        .def("HasConfig", &ConfigScope::HasConfig, nb::arg("key"))
        .def("Type",
            [](const ConfigScope &scope, const std::string &key) -> std::string {
                return scope.Type(key).name();
            },
            nb::arg("key"))
        .def("ToString", &ConfigScope::ToString);

    nb::class_<CubeTile>(m, "CubeTile")
    .def(nb::init<>())
    .def(nb::init<const std::array<int64_t, MAX_M_DIM_SIZE>&,
                   const std::array<int64_t, MAX_K_DIM_SIZE>&,
                   const std::array<int64_t, MAX_N_DIM_SIZE>&,
                   bool, bool>(),
         nb::arg("m"),
         nb::arg("k"),
         nb::arg("n"),
         nb::arg("enableMultiDataLoad") = false,
         nb::arg("enableSplitK") = false)
    .def_rw("m", &CubeTile::m)
    .def_rw("k", &CubeTile::k)
    .def_rw("n", &CubeTile::n)
    .def_rw("enableMultiDataLoad", &CubeTile::enableMultiDataLoad)
    .def_rw("enableSplitK", &CubeTile::enableSplitK)
    .def("valid", &CubeTile::valid)
    .def("ToString", &CubeTile::ToString)
    .def("__repr__", [](const CubeTile &t) { return t.ToString(); })
    .def("__str__",  [](const CubeTile &t) { return t.ToString(); });
}

void BindController(nb::module_ &m) {
    bind_controller_config(m);
    bind_controller_set_tile(m);
    bind_controller_function(m);
    bind_controller_loop(m);
    bind_controller_utils(m);
    bind_controller_scope(m);
    bind_controller_scope_classes(m);

    // disable cpp mode
    SourceLocation::SetCppMode(false);
}
} // namespace pypto
