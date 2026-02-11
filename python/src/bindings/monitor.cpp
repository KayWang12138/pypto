/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file monitor.cpp
 * \brief Python bindings for compiler monitoring
 */

#include "pybind_common.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_config.h"

using namespace npu::tile_fwk;

namespace pypto {

void SetMonitorOptions(bool enable, int intervalSec, int timeoutSec, int totalTimeoutSec, int timeoutAction) {
    auto action = static_cast<TimeoutAction>(timeoutAction);

    if (action != TimeoutAction::THROW_EXCEPTION && action != TimeoutAction::WARN_ONLY) {
        throw std::invalid_argument("timeoutAction must be 0 (THROW_EXCEPTION) or 1 (WARN_ONLY)");
    }

    if (totalTimeoutSec < 0) {
        throw std::invalid_argument("totalTimeoutSec must be >= 0 (0 = disabled)");
    }

    MonitorManager::Instance().SetOptions(enable, intervalSec, timeoutSec, totalTimeoutSec, action);
}

void SetMonitorStageMode(int mode, const std::string& customStagesStr) {
    // Validate mode
    if (mode < 0 || mode > 2) {
        throw std::invalid_argument("mode must be 0 (COARSE), 1 (FINE), or 2 (CUSTOM)");
    }

    auto stageMode = static_cast<StageConfig::Mode>(mode);

    std::vector<std::string> customStages;
    if (stageMode == StageConfig::Mode::CUSTOM) {
        // Parse comma-separated stage names
        std::stringstream ss(customStagesStr);
        std::string stage;
        while (std::getline(ss, stage, ',')) {
            // Trim whitespace
            stage.erase(0, stage.find_first_not_of(" \t\n\r"));
            stage.erase(stage.find_last_not_of(" \t\n\r") + 1);
            if (!stage.empty()) {
                customStages.push_back(stage);
            }
        }

        if (customStages.empty()) {
            throw std::invalid_argument("custom_stages must not be empty when mode=CUSTOM");
        }
    }

    StageConfig::Instance().LoadConfig(stageMode, customStages);
}

void InitializeMonitor() {
    MonitorManager::Instance().Initialize();
}

void ShutdownMonitor() {
    MonitorManager::Instance().Shutdown();
}

void BindMonitor(py::module &m) {
    // Bind monitor configuration functions
    m.def("SetMonitorOptions", &SetMonitorOptions,
          py::arg("enable"), py::arg("interval_sec"), py::arg("timeout_sec"), py::arg("total_timeout_sec") = 0, py::arg("timeout_action"),
          "Set compiler monitor options.\n"
          "Args:\n"
          "    enable: Whether to enable monitoring\n"
          "    interval_sec: Progress print interval in seconds\n"
          "    timeout_sec: Per-stage timeout threshold in seconds\n"
          "    total_timeout_sec: Total compilation timeout threshold in seconds (default: 0 = disabled)\n"
          "    timeout_action: 0=THROW_EXCEPTION, 1=WARN_ONLY");

    m.def("SetMonitorStageMode", &SetMonitorStageMode,
          py::arg("mode"), py::arg("custom_stages") = "",
          "Set compiler stage mode.\n"
          "Args:\n"
          "    mode: 0=COARSE (6 major stages), 1=FINE (individual passes), 2=CUSTOM\n"
          "    custom_stages: Comma-separated custom stage names (required when mode=2)");

    m.def("InitializeMonitor", &InitializeMonitor,
          "Initialize the compiler monitor. Must be called before compilation.");

    m.def("ShutdownMonitor", &ShutdownMonitor,
          "Shutdown the compiler monitor and release resources.");

    // Bind TimeoutAction enum
    py::enum_<TimeoutAction>(m, "TimeoutAction")
        .value("THROW_EXCEPTION", TimeoutAction::THROW_EXCEPTION)
        .value("WARN_ONLY", TimeoutAction::WARN_ONLY)
        .export_values();

    // Bind StageConfig::Mode enum
    py::enum_<StageConfig::Mode>(m, "StageMode")
        .value("COARSE", StageConfig::Mode::COARSE)
        .value("FINE", StageConfig::Mode::FINE)
        .value("CUSTOM", StageConfig::Mode::CUSTOM)
        .export_values();

    // Bind predefined coarse stage constants
    m.attr("COARSE_STAGES") = std::vector<std::string>{
        "FrontendParser",
        "TensorGraphPass",
        "TileGraphPass",
        "BlockGraphPass",
        "CodeGen",
        "BinaryGeneration"
    };
}

} // namespace pypto
