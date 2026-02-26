/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pybind_common.h"
#include "interface/compiler_monitor/monitor_manager.h"

using namespace npu::tile_fwk;

namespace pypto {

void BindMonitor(py::module &m) {
    m.def("InitializeMonitor", []() {
        MonitorManager::Instance().Initialize();
    }, "Initialize compiler monitor (called on pypto import).");

    m.def("SetCompilerMonitorOptions", [](bool enable, int interval_sec, int timeout_sec, int total_timeout_sec) {
        MonitorManager::Instance().SetCompilerMonitorOptions(enable, interval_sec, timeout_sec, total_timeout_sec);
    }, py::arg("enable") = true, py::arg("interval_sec") = 30, py::arg("timeout_sec") = -1,
        py::arg("total_timeout_sec") = 600,
        "Set compiler monitor options: enable, progress print interval in seconds.");
}

}  // namespace pypto
