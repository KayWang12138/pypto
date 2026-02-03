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
 * \file pass_config.cpp
 * \brief
 */

#include "nb_common.h"

using namespace npu::tile_fwk;

namespace pypto {
void bind_pass_const(nb::module_ &m) {
    m.attr("KEY_DUMP_GRAPH") = KEY_DUMP_GRAPH;
    m.attr("KEY_PRINT_GRAPH") = KEY_PRINT_GRAPH;
    m.attr("KEY_PRE_CHECK") = KEY_PRE_CHECK;
    m.attr("KEY_POST_CHECK") = KEY_POST_CHECK;
    m.attr("KEY_HEALTH_CHECK") = KEY_HEALTH_CHECK;
}

// config pass_global_configs.x parameters
void bind_pass_global_config(nb::module_ &m) {
    m.def("GetPassGlobalConfig", [](const std::string &key, const bool &default_value) -> nb::object {
            return nb::cast(config::GetPassGlobalConfig<bool>(key, default_value));
        }, nb::arg("key"), nb::arg("default_value"));
    m.def("GetPassGlobalConfig", [](const std::string &key, const int64_t &default_value) -> nb::object {
            return nb::cast(config::GetPassGlobalConfig<int64_t>(key, default_value));
        }, nb::arg("key"), nb::arg("default_value"));
    m.def("GetPassGlobalConfig", [](const std::string &key, const std::string &default_value) -> nb::object {
            return nb::cast(config::GetPassGlobalConfig<std::string>(key, default_value));
        }, nb::arg("key"), nb::arg("default_value"));

    m.def("SetPassGlobalConfig", [](const std::string &key, const bool &value) {
            config::SetPassGlobalConfig<bool>(key, value);
        }, nb::arg("key"), nb::arg("value"));
    m.def("SetPassGlobalConfig", [](const std::string &key, const int64_t &value) {
            config::SetPassGlobalConfig<int64_t>(key, value);
        }, nb::arg("key"), nb::arg("value"));
    m.def("SetPassGlobalConfig", [](const std::string &key, const std::string &value) {
            config::SetPassGlobalConfig<std::string>(key, value);
        }, nb::arg("key"), nb::arg("value"));
}

// config pass_global_configs.default_pass_configs.x parameters
void bind_pass_default_config(nb::module_ &m) {
    m.def("GetPassDefaultConfig", [](const std::string &key, const bool &default_value) -> nb::object {
            return nb::cast(config::GetPassDefaultConfig<bool>(key, default_value));
        }, nb::arg("key"), nb::arg("default_value"));
    m.def("GetPassDefaultConfig", [](const std::string &key, const int64_t &default_value) -> nb::object {
            return nb::cast(config::GetPassDefaultConfig<int64_t>(key, default_value));
        }, nb::arg("key"), nb::arg("default_value"));
    m.def("GetPassDefaultConfig", [](const std::string &key, const std::string &default_value) -> nb::object {
            return nb::cast(config::GetPassDefaultConfig<std::string>(key, default_value));
        }, nb::arg("key"), nb::arg("default_value"));

    m.def("SetPassDefaultConfig", [](const std::string &key, const bool &value) {
            config::SetPassDefaultConfig<bool>(key, value);
        }, nb::arg("key"), nb::arg("value"));
    m.def("SetPassDefaultConfig", [](const std::string &key, const int64_t &value) {
            config::SetPassDefaultConfig<int64_t>(key, value);
        }, nb::arg("key"), nb::arg("value"));
    m.def("SetPassDefaultConfig", [](const std::string &key, const std::string &value) {
            config::SetPassDefaultConfig<std::string>(key, value);
        }, nb::arg("key"), nb::arg("value"));
}

// config strategies.x parameters
void bind_pass_config(nb::module_ &m) {
    m.def("GetPassConfig",
        [](const std::string &strategy, const std::string &identifier, const std::string &key, const bool &default_value) -> nb::object {
            return nb::cast(config::GetPassConfig<bool>(strategy, identifier, key, default_value));
        }, nb::arg("strategy"), nb::arg("identifier"), nb::arg("key"), nb::arg("default_value"));
    m.def("GetPassConfig",
        [](const std::string &strategy, const std::string &identifier, const std::string &key, const int64_t &default_value) -> nb::object {
            return nb::cast(config::GetPassConfig<int64_t>(strategy, identifier, key, default_value));
        }, nb::arg("strategy"), nb::arg("identifier"), nb::arg("key"), nb::arg("default_value"));
    m.def("GetPassConfig",
        [](const std::string &strategy, const std::string &identifier, const std::string &key, const std::string &default_value) -> nb::object {
            return nb::cast(config::GetPassConfig<std::string>(strategy, identifier, key, default_value));
        }, nb::arg("strategy"), nb::arg("identifier"), nb::arg("key"), nb::arg("default_value"));

    m.def("SetPassConfig",
        [](const std::string &strategy, const std::string &identifier, const std::string &key, const bool &value) {
            config::SetPassConfig<bool>(strategy, identifier, key, value);
        }, nb::arg("strategy"), nb::arg("identifier"), nb::arg("key"), nb::arg("value"));
    m.def("SetPassConfig",
        [](const std::string &strategy, const std::string &identifier, const std::string &key, const int64_t &value) {
            config::SetPassConfig<int64_t>(strategy, identifier, key, value);
        }, nb::arg("strategy"), nb::arg("identifier"), nb::arg("key"), nb::arg("value"));
    m.def("SetPassConfig",
        [](const std::string &strategy, const std::string &identifier, const std::string &key, const std::string &value) {
            config::SetPassConfig<std::string>(strategy, identifier, key, value);
        }, nb::arg("strategy"), nb::arg("identifier"), nb::arg("key"), nb::arg("value"));
}

void bind_pass_configs(nb::module_ &m) {
    nb::class_<PassConfigs>(m, "PassConfigs")
        .def_ro("printGraph", &PassConfigs::printGraph)
        .def_ro("dumpGraph", &PassConfigs::dumpGraph)
        .def_ro("dumpPassTimeCost", &PassConfigs::dumpPassTimeCost)
        .def_ro("preCheck", &PassConfigs::preCheck)
        .def_ro("postCheck", &PassConfigs::postCheck)
        .def_ro("disablePass", &PassConfigs::disablePass)
        .def_ro("healthCheck", &PassConfigs::healthCheck);
    m.def("GetPassConfigs",
        [](const std::string &strategy, const std::string &identifier) -> PassConfigs {
            return ConfigManager::Instance().GetPassConfigs(strategy, identifier);
        }, nb::arg("strategy"), nb::arg("identifier"));
}

void BindPass(nb::module_ &m) {
    bind_pass_const(m);
    bind_pass_global_config(m);
    bind_pass_default_config(m);
    bind_pass_config(m);
    bind_pass_configs(m);
    // disable cpp mode
    SourceLocation::SetCppMode(false);
}
} // namespace pypto
